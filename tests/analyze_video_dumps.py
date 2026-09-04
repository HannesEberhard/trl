"""
Diagnostic script for comparing GRPO video debug dumps (TRL_DEBUG_VIDEO_DUMP_DIR).

A plain `torch.equal(client["pixel_values_videos"], server_side["pixel_values_videos"])` conflates several
independent things that can each cause a mismatch:
  1. dtype: vLLM's own multimodal processing casts pixel values to the model's runtime dtype (e.g. bfloat16)
     as a post-processing step; a bare `processor(...)` call on the client stays in the processor's native
     dtype (usually float32). This alone makes `torch.equal` return False even when the underlying float32
     math was identical.
  2. batch alignment: `client_loss_forward_*.pt`'s `pixel_values_videos` is a BATCHED tensor covering every
     video in that training step (multiple prompts x num_generations), not a single video. Comparing it
     directly against a single reconstructed video's tensor is a shape/content mismatch by construction.
  3. wire transport: whether the base64-PNG-encoded frames the server received are byte-identical to what the
     client sent, independent of any pixel-value computation.

This script checks each of these in isolation:
  Stage A - wire transport: are the raw frames the server received pixel-identical to what the client sent?
  Stage B - pixel-value computation: given the SAME frames+metadata, does calling `processor(...)` directly
            reproduce (up to dtype) the slice of `client_loss_forward`'s `pixel_values_videos` that
            corresponds to one video?
  Stage C - input IDs: does the exact token sequence sent to vLLM match what the loss-time re-tokenization of
            the same prompt text produces?
  Stage D - placeholder-token scatter alignment: does vLLM's own internal prompt-placeholder-matching machinery
            (`BaseMultiModalProcessor.apply()` -> `_maybe_apply_prompt_updates`) leave the prompt TRL sent
            unchanged, or does it re-match/mutate it? TRL sends an *already fully expanded* prompt (real
            per-frame timestamps + N soft-tokens per frame, via `apply_chat_template(tokenize=True, ...)`) —
            but vLLM's placeholder-matching code is built to find a single collapsed placeholder token per
            item and expand it itself. Requires the debug patch in `vllm/multimodal/processing/processor.py`
            (see `TRL_DEBUG_VLLM_PLACEHOLDER_DUMP_DIR` below) to be applied to your vLLM checkout and the
            server run with that env var set; pass its dump dir via `--vllm_dump_dir`.

Usage:
    python tests/analyze_video_dumps.py --dump_dir /tmp/trl_video_debug --model google/gemma-4-12B-it
    # add --vllm_dump_dir /tmp/vllm_placeholder_debug for Stage D (requires the vLLM-side debug patch)

Tip: clear the dump dir before a run (`rm -rf /tmp/trl_video_debug/*`) and use `--steps 1` on the smoke test
(`tests/video_grpo_smoke_test.py`), so "latest file per prefix" unambiguously refers to the same rollout on
both the client and server side.
"""

import argparse
import glob
import os

import torch
from transformers import AutoProcessor


def load_latest(dump_dir: str, prefix: str):
    files = sorted(glob.glob(os.path.join(dump_dir, f"{prefix}*.pt")))
    if not files:
        raise FileNotFoundError(f"No files matching {prefix}*.pt in {dump_dir}")
    print(f"[{prefix}] found {len(files)} file(s), using: {files[-1]}")
    return torch.load(files[-1], weights_only=False), files[-1]


def frames_equal(frames_a, frames_b) -> bool:
    if len(frames_a) != len(frames_b):
        print(f"  frame count differs: {len(frames_a)} vs {len(frames_b)}")
        return False
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b, strict=True)):
        if list(fa.getdata()) != list(fb.getdata()):
            print(f"  frame {i} differs pixel-for-pixel")
            return False
    return True


def stage_a_wire_transport(dump_dir: str):
    print("\n=== Stage A: wire transport (client-sent vs server-received frames) ===")
    client_req, _ = load_latest(dump_dir, "client_rollout_request")
    server_req, _ = load_latest(dump_dir, "server_generate_request")

    # client_req = {"videos": [[frames_per_video, ...] per prompt], "video_metadata": [...], "prompt_ids": [...]}
    # server_req = [ [ (frames, metadata), ... ] per prompt-with-video ]  (see vllm_serve.py's dump hook)
    client_videos = client_req["videos"]
    client_metadata = client_req["video_metadata"]

    # Flatten client side to a flat list of (frames, metadata), matching server_req's flattening.
    client_flat = []
    for prompt_videos, prompt_meta in zip(client_videos, client_metadata, strict=True):
        if not prompt_videos:
            continue
        for frames, meta in zip(prompt_videos, prompt_meta, strict=True):
            client_flat.append((frames, meta))
    server_flat = [video for prompt_videos in server_req for video in prompt_videos]

    print(f"client sent {len(client_flat)} video(s) with video content; server received {len(server_flat)}")
    if len(client_flat) != len(server_flat):
        print("!! COUNT MISMATCH — likely comparing dumps from different steps/requests. Re-check file selection.")
        return

    all_ok = True
    for i, ((c_frames, c_meta), (s_frames, s_meta)) in enumerate(zip(client_flat, server_flat, strict=True)):
        ok = frames_equal(c_frames, s_frames)
        meta_ok = dict(c_meta) == dict(s_meta) if c_meta and s_meta else (c_meta == s_meta)
        print(f"video {i}: frames_identical={ok}, metadata_identical={meta_ok}")
        if not meta_ok:
            print(f"  client metadata: {c_meta}")
            print(f"  server metadata: {s_meta}")
        all_ok = all_ok and ok and meta_ok

    print("Stage A result:", "PASS — wire transport is lossless" if all_ok else "FAIL — see above")
    return client_flat


def stage_b_pixel_computation(dump_dir: str, model_id: str, reference_frames_metadata):
    print("\n=== Stage B: pixel-value computation (processor(...) vs client's dumped pixel_values_videos) ===")
    client_dump, client_path = load_latest(dump_dir, "client_loss_forward")

    pixel_key = "pixel_values_videos"
    if pixel_key not in client_dump or client_dump[pixel_key] is None:
        print(f"No `{pixel_key}` in {client_path} — nothing to compare (check the file actually has video data).")
        return

    client_pixel_values = client_dump[pixel_key]
    print(f"client pixel_values_videos: shape={tuple(client_pixel_values.shape)}, dtype={client_pixel_values.dtype}")

    grid_key = "video_grid_thw" if client_dump.get("video_grid_thw") is not None else "video_position_ids"
    grid_tensor = client_dump.get(grid_key)
    print(f"client {grid_key}: {grid_tensor.shape if grid_tensor is not None else None}")

    processor = AutoProcessor.from_pretrained(model_id)

    if reference_frames_metadata is None:
        print("No Stage A reference available; skipping Stage B (nothing to recompute from).")
        return
    frames, metadata = reference_frames_metadata[0]  # first video; all videos are identical in this smoke test

    # Recompute pixel values directly, exactly mirroring the loss-time call in grpo_trainer.py
    # (`self.processing_class(videos=..., video_metadata=..., do_sample_frames=False, text=..., ...)`).
    recomputed = processor(
        videos=[frames],
        video_metadata=[metadata],
        do_sample_frames=False,
        text=["placeholder"],
        return_tensors="pt",
    )
    recomputed_pixel_values = recomputed[pixel_key]
    print(f"recomputed pixel_values_videos: shape={tuple(recomputed_pixel_values.shape)}, "
          f"dtype={recomputed_pixel_values.dtype}")

    # Slice the corresponding video out of the client's batched tensor.
    if grid_key == "video_position_ids" or grid_tensor is None:
        # Gemma-style: one row per video, indexed directly.
        client_slice = client_pixel_values[: recomputed_pixel_values.shape[0]]
    else:
        # Qwen-style: rows determined by video_grid_thw's product per video.
        rows = grid_tensor[0].prod().item()
        client_slice = client_pixel_values[:rows]

    print(f"comparing client_slice {tuple(client_slice.shape)} vs recomputed {tuple(recomputed_pixel_values.shape)}")
    if client_slice.shape != recomputed_pixel_values.shape:
        print("!! SHAPE MISMATCH — slicing logic above is likely wrong for this model; inspect shapes manually.")
        return

    # Raw (dtype-mismatched) comparison first, to show the false alarm dtype causes.
    same_dtype = client_slice.dtype == recomputed_pixel_values.dtype
    print(f"dtypes match: {same_dtype} (client={client_slice.dtype}, recomputed={recomputed_pixel_values.dtype})")
    if client_slice.device != recomputed_pixel_values.device:
        print(f"devices differ (client={client_slice.device}, recomputed={recomputed_pixel_values.device}); "
              "moving both to CPU for comparison.")
        client_slice = client_slice.cpu()
        recomputed_pixel_values = recomputed_pixel_values.cpu()

    recomputed_cast = recomputed_pixel_values.to(client_slice.dtype)
    exact = torch.equal(client_slice, recomputed_cast)
    close = torch.allclose(client_slice.float(), recomputed_cast.float(), atol=1e-3, rtol=1e-3)
    max_abs_diff = (client_slice.float() - recomputed_cast.float()).abs().max().item()

    print(f"torch.equal (after dtype cast): {exact}")
    print(f"torch.allclose (atol=1e-3): {close}")
    print(f"max abs diff: {max_abs_diff}")

    if exact:
        print("Stage B result: PASS — pixel-value computation is bit-identical once dtype is aligned.")
    elif close:
        print("Stage B result: CLOSE — small numeric diff, likely dtype-cast rounding, not a correctness bug.")
    else:
        print("Stage B result: FAIL — genuine divergence, worth digging into further (see max abs diff above).")


def stage_c_input_ids(dump_dir: str, model_id: str):
    """
    Compares the exact token IDs sent to vLLM (`client_rollout_request`'s `prompt_ids`) against the token IDs the
    loss-time forward pass actually uses (`client_loss_forward`'s `input_ids`).

    Post the placeholder-expansion-mismatch fix (see VIDEO_SUPPORT_HANDOFF.md §8b), these are EXPECTED TO DIFFER
    for video prompts: TRL now sends vLLM a short, collapsed-placeholder prompt (`vllm_prompt_ids`, built via
    `_build_vllm_video_prompt_ids` in `grpo_trainer.py`) so vLLM's own placeholder-matching machinery can expand
    it correctly itself, while the loss-time forward pass keeps using the separately-computed, already fully-
    expanded `prompt_ids`. A length/content mismatch here is no longer a bug signal for video — it's Stage D's
    job to verify the two forms are still *consistent* (does vLLM's own expansion of the short form reproduce
    the loss-time full form exactly?). For text-only or image-only prompts (not yet covered by this fix), the
    two should still match, so a mismatch there IS worth investigating.

    Note: `client_rollout_request`'s `prompt_ids` is the deduplicated (one row per unique prompt) list actually
    sent over the wire; `client_loss_forward`'s `input_ids` is the full, padded, non-deduplicated training batch
    (num_generations copies included). This compares the FIRST row of each — valid as an exact check only when
    row 0 of both corresponds to the same prompt (true here since every example in this smoke test is identical;
    for a real dataset with varying prompts, match by decoding and comparing text instead of assuming row 0).
    """
    print("\n=== Stage C: input IDs (client-sent to vLLM vs loss-time forward pass) ===")
    rollout, _ = load_latest(dump_dir, "client_rollout_request")
    loss_fwd, _ = load_latest(dump_dir, "client_loss_forward")

    sent_prompt_ids = rollout.get("prompt_ids")
    input_ids = loss_fwd.get("input_ids")
    attention_mask = loss_fwd.get("attention_mask")
    if sent_prompt_ids is None or input_ids is None:
        print("Missing `prompt_ids` or `input_ids` in the dumps — re-run the smoke test with the latest code.")
        return

    sent_ids_0 = list(sent_prompt_ids[0])
    mask_0 = attention_mask[0].bool() if attention_mask is not None else torch.ones_like(input_ids[0]).bool()
    loss_ids_0 = input_ids[0][mask_0].tolist()  # strip padding regardless of padding side

    print(f"sent to vLLM      ({len(sent_ids_0)} tokens): {sent_ids_0}")
    print(f"loss-time forward ({len(loss_ids_0)} tokens): {loss_ids_0}")

    # Decode to text unconditionally (not just on mismatch) so the actual rendered prompt — video placeholder
    # tokens, timestamps, chat-template formatting — can be eyeballed even when the IDs already match exactly.
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(processor, "tokenizer", processor)
        print("sent to vLLM      (text):", tokenizer.decode(sent_ids_0))
        print("loss-time forward (text):", tokenizer.decode(loss_ids_0))
    except Exception as e:  # decoding is best-effort diagnostics, not required for the comparison itself
        print(f"(could not decode for display: {e})")

    if sent_ids_0 == loss_ids_0:
        print("Stage C result: PASS — identical token sequence (expected for text/image-only prompts).")
    elif len(sent_ids_0) < len(loss_ids_0):
        print(
            "Stage C result: EXPECTED DIFFERENCE — sent-to-vLLM is shorter than loss-time, consistent with the "
            "short-collapsed-placeholder fix for video (see docstring above). Check Stage D for the real "
            "consistency guarantee, not this stage."
        )
    else:
        print("Stage C result: FAIL — sequences differ in an unexpected way (sent-to-vLLM is not shorter); investigate.")


def stage_d_placeholder_alignment(dump_dir: str, vllm_dump_dir: str):
    """
    Checks vLLM's own placeholder expansion (captured by the debug patch to
    `vllm/multimodal/processing/processor.py`'s `apply()`, env var `TRL_DEBUG_VLLM_PLACEHOLDER_DUMP_DIR` set on
    the `trl vllm-serve` process) against the independently-computed, fully-expanded prompt TRL uses for the
    loss-time forward pass (`client_loss_forward`'s `input_ids`).

    Post the placeholder-expansion-mismatch fix (see VIDEO_SUPPORT_HANDOFF.md §8b): TRL now sends vLLM a short,
    collapsed-placeholder prompt (`client_rollout_request`'s `prompt_ids`) instead of an already-expanded one.
    vLLM is *expected* to expand it (input != output is now normal, not a bug). The real correctness question
    is whether vLLM's own expansion produces the *exact same* token sequence TRL's own `_tokenize_prompts`
    independently computed for the loss-time forward pass — i.e. do the two places that each build the "real"
    prompt (vLLM's placeholder-matching machinery, and TRL's `apply_chat_template(tokenize=True, ...)` call)
    agree with each other, byte-for-byte? If they don't, generation and training are still conditioned on
    different contexts, even though neither one is "the corrupted double-length prompt" anymore.
    """
    print("\n=== Stage D: placeholder expansion consistency (vLLM's own expansion vs TRL's loss-time prompt) ===")
    rollout, _ = load_latest(dump_dir, "client_rollout_request")
    sent_prompt_ids = rollout.get("prompt_ids")
    loss_fwd, _ = load_latest(dump_dir, "client_loss_forward")
    input_ids = loss_fwd.get("input_ids")
    attention_mask = loss_fwd.get("attention_mask")
    if sent_prompt_ids is None or input_ids is None:
        print("Missing `prompt_ids` or `input_ids` in the dumps — re-run the smoke test with the latest code.")
        return

    try:
        vllm_dump, vllm_path = load_latest(vllm_dump_dir, "apply_prompt_updates")
    except FileNotFoundError as e:
        print(f"{e}\nDid you apply the vLLM debug patch and set TRL_DEBUG_VLLM_PLACEHOLDER_DUMP_DIR on the server?")
        return

    sent_ids_0 = list(sent_prompt_ids[0])
    vllm_input_ids = vllm_dump["input_prompt_ids"]
    vllm_output_ids = vllm_dump["output_prompt_ids"]
    mask_0 = attention_mask[0].bool() if attention_mask is not None else torch.ones_like(input_ids[0]).bool()
    loss_ids_0 = input_ids[0][mask_0].tolist()

    print(f"TRL sent to vLLM (short form)   ({len(sent_ids_0)} tokens)")
    print(f"vLLM apply() saw as input       ({vllm_dump['input_len']} tokens)")
    print(f"vLLM apply() produced as output ({vllm_dump['output_len']} tokens)")
    print(f"TRL's loss-time full prompt     ({len(loss_ids_0)} tokens)")
    print(f"is_update_applied: {vllm_dump['is_update_applied']}")

    if sent_ids_0 != vllm_input_ids:
        print(
            "!! `client_rollout_request`'s prompt_ids don't match apply()'s recorded input — likely comparing "
            "dumps from different requests/steps, or per-item vs batched request shape. Re-check file selection."
        )

    expanded = vllm_input_ids != vllm_output_ids
    print(f"vLLM expanded the short prompt: {expanded} (expected: True, post-fix)")

    consistent = vllm_output_ids == loss_ids_0
    print(f"vLLM's expansion matches TRL's loss-time prompt exactly: {consistent}")
    if not consistent:
        print(
            f"  Length diff: vLLM output {vllm_dump['output_len']} vs loss-time {len(loss_ids_0)} "
            f"({vllm_dump['output_len'] - len(loss_ids_0):+d})"
        )
        print(f"  mm_placeholder_ranges: {vllm_dump['mm_placeholder_ranges']}")

    if consistent:
        print("Stage D result: PASS — vLLM's own placeholder expansion exactly matches TRL's loss-time prompt.")
    else:
        print("Stage D result: FAIL — vLLM's expansion diverges from TRL's loss-time prompt; see details above.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", default="/tmp/trl_video_debug")
    parser.add_argument(
        "--vllm_dump_dir",
        default=None,
        help="Dir set via TRL_DEBUG_VLLM_PLACEHOLDER_DUMP_DIR on the vllm-serve process, for Stage D.",
    )
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    reference = stage_a_wire_transport(args.dump_dir)
    stage_b_pixel_computation(args.dump_dir, args.model, reference)
    stage_c_input_ids(args.dump_dir, args.model)
    if args.vllm_dump_dir:
        stage_d_placeholder_alignment(args.dump_dir, args.vllm_dump_dir)


if __name__ == "__main__":
    main()
