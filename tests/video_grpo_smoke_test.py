"""
Minimal end-to-end smoke test for GRPO + vllm_mode="server", across text/image/video modalities.

Usage:
    # Terminal 1 (uses e.g. GPU 0 for the vLLM server):
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model <MODEL_ID>

    # Terminal 2 (uses the remaining GPU(s) for training). Must be wrapped in `accelerate launch` — running this
    # as plain `python ...` leaves `Accelerator().device` without an explicit CUDA index, which makes vLLM's
    # `PyNcclCommunicator` weight-sync self-test fail with
    # "this nccl communicator is created to work on cuda, but the input tensor is on cuda:0".
    CUDA_VISIBLE_DEVICES=1 TRL_DEBUG_VIDEO_DUMP_DIR=/tmp/trl_video_debug accelerate launch --num_processes=1 video_grpo_smoke_test.py --model <MODEL_ID> --modality video

`--modality` selects the dataset built for the run: `text` (no visual input, useful as a control — e.g. to check
whether an `importance_sampling_ratio` anomaly is modality-specific or general vLLM-vs-transformers drift),
`image`, or `video` (default).

After it finishes, list what got dumped and inspect one:
    ls -la /tmp/trl_video_debug
    python -c "
import torch
d = torch.load('/tmp/trl_video_debug/<pick-a-client_loss_forward-file>.pt', weights_only=False)
for k, v in d.items():
    print(k, getattr(v, 'shape', v))
"
`client_rollout_request_*.pt` / `client_loss_forward_*.pt` come from the trainer process (see
`GRPOTrainer._maybe_dump_video_debug_tensors`); `server_generate_request_*.pt` comes from the vLLM server process
(see the debug hook in `trl/scripts/vllm_serve.py`). The client-side dump for `client_loss_forward` already
contains the final `pixel_values_videos` tensor; the server-side dump contains the raw (frames, metadata) vLLM
received, so to compare pixel values against it you'd re-run it through the same processor call, e.g.:
    processor(videos=[frames], video_metadata=[metadata], do_sample_frames=False, text=[...], return_tensors="pt")
"""

import argparse
import base64
import io

from datasets import Dataset
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer


def make_synthetic_video_data_url(num_frames: int = 40, size: int = 64, fps: int = 4) -> str:
    """
    Encode a tiny synthetic video (solid color frames) as an mp4, return it as a base64 data URL.

    `num_frames` defaults to 40, comfortably above Gemma4's video processor default of `num_frames=32`: its
    `sample_frames` raises if asked to sample more frames than the video actually has (no frame repeats).
    Qwen2-VL/2.5-VL don't have this constraint (they sample by `fps`, not by defaulting to a fixed frame count),
    but 40 frames works for either.
    """
    import av
    import numpy as np

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        color = int(255 * i / num_frames)
        arr = np.full((size, size, 3), (color, 0, 255 - color), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:video/mp4;base64,{payload}"


def make_synthetic_image(size: int = 64):
    """A single solid-color PIL image, matching one frame of `make_synthetic_video_data_url`'s style."""
    from PIL import Image

    return Image.new("RGB", (size, size), color=(128, 0, 128))


def build_dataset(modality: str, num_examples: int = 8) -> Dataset:
    if modality == "text":
        prompt = [{"role": "user", "content": "Write one sentence about the weather."}]
        return Dataset.from_dict({"prompt": [prompt] * num_examples})
    elif modality == "image":
        prompt = [{"role": "user", "content": "Describe what happens in this image in one sentence."}]
        return Dataset.from_dict(
            {"prompt": [prompt] * num_examples, "image": [make_synthetic_image() for _ in range(num_examples)]}
        )
    elif modality == "video":
        prompt = [{"role": "user", "content": "Describe what happens in this video in one sentence."}]
        video_url = make_synthetic_video_data_url()
        return Dataset.from_dict({"prompt": [prompt] * num_examples, "video": [video_url] * num_examples})
    else:
        raise ValueError(f"Unknown modality: {modality!r}. Expected 'text', 'image', or 'video'.")


def reward_len(completions, **kwargs):
    return [float(len(c[0]["content"])) for c in completions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--modality",
        choices=["text", "image", "video"],
        default="video",
        help="Which dataset/input type to train on. 'text' is a useful control run (no visual input).",
    )
    parser.add_argument("--output_dir", default="./trl_video_smoke_test_out")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--no_peft", action="store_true", help="Disable LoRA and train the full model.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names to attach LoRA adapters to.",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.modality)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://piora1:8000",
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=32,
        max_steps=args.steps,
        learning_rate=1e-6,
        report_to="none",
        logging_steps=1,
    )
    # LoRA cuts optimizer-state and gradient memory (the base weights are still loaded in full precision, so this
    # alone won't help if the OOM is from just loading a 12B model — add `--load_in_4bit` via a quantization_config
    # on top of this if it's still tight).
    peft_config = (
        None
        if args.no_peft
        else LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            task_type="CAUSAL_LM",
        )
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    print(f"Training completed successfully. modality={args.modality}")


if __name__ == "__main__":
    main()
