"""
Standalone, vLLM-independent reproduction of a suspected bug in how vLLM's
prompt-placeholder-matching machinery handles a prompt that TRL has *already*
fully expanded (real per-frame timestamps + N soft-tokens per frame), instead
of the short, collapsed single-placeholder-per-item prompt this matching code
appears to be designed for.

Does NOT import vllm. `iter_token_matches` and `_apply_matches` below are
copied verbatim (only renamed slightly / trimmed of type-only imports) from
vllm's vllm/multimodal/processing/processor.py, lines 619-672 and 700-828, so
this exercises vLLM's *actual* algorithm, not a paraphrase of it.

Run: python tests/repro_vllm_placeholder_mismatch.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Copied verbatim from vllm/multimodal/processing/processor.py:619-672
# ---------------------------------------------------------------------------


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int


def iter_token_matches(token_ids, match_ids, *, start_idx=0):
    """Yield each occurrence of `match_ids` in `token_ids`. (verbatim from vLLM)"""
    prompt_len = len(token_ids)
    match_len = len(match_ids)

    if match_len == 0:
        return

    while start_idx < prompt_len - match_len + 1:
        end_idx = start_idx + match_len

        if token_ids[start_idx:end_idx] == match_ids:
            yield _TokenMatch(start_idx=start_idx, end_idx=end_idx)

            # Exclude overlapping matches
            start_idx = end_idx
        else:
            start_idx += 1


# ---------------------------------------------------------------------------
# Minimal stand-ins for vLLM's PromptTargetMatch / UpdateMode / the parts of
# _find_matches + _apply_matches (processor.py:700-828) needed to exercise
# the real per-item single-token matching + "only replace one non-empty item
# at a time" logic, verbatim in spirit. Simplified to a single modality
# ("video") since that's all this repro needs.
# ---------------------------------------------------------------------------


class PromptTargetMatch(NamedTuple):
    start_idx: int
    end_idx: int


@dataclass
class VideoUpdate:
    """Stand-in for one item's ResolvedPromptUpdate (mode is always REPLACE)."""

    item_idx: int
    target: list[int]  # the short, collapsed placeholder to search for
    replacement: list[int]  # the full, already-correct per-video expansion


def find_matches_for_item(prompt: list[int], update: VideoUpdate, start_idx: int):
    """Mirrors ResolvedPromptUpdate.iter_token_matches (processor.py:547-567): find
    the first occurrence of `update.target` in `prompt`, starting at `start_idx`."""
    for m in iter_token_matches(prompt, update.target, start_idx=start_idx):
        return PromptTargetMatch(m.start_idx, m.end_idx)
    return None


def apply_matches(prompt: list[int], updates: list[VideoUpdate]) -> tuple[list[int], dict[int, PromptTargetMatch]]:
    """
    Mirrors _apply_matches (processor.py:765-828), specialized to a single
    modality ("video") with mode=REPLACE for every item. Per real vLLM: only
    ONE non-empty match is actually applied per outer-loop pass
    (processor.py:737-750, "To avoid conflicts, only replace one non-empty
    item at a time"), then the loop repeats with prev_end_idx advanced past
    just that one consumed match.
    """
    resolved: dict[int, PromptTargetMatch] = {}
    out_chunks: list[list[int]] = []
    prev_end_idx = 0

    while True:
        # _find_matches: for every not-yet-resolved item, find its first match
        # from prev_end_idx onward (processor.py:711-732).
        candidates: dict[int, PromptTargetMatch] = {}
        for update in updates:
            if update.item_idx in resolved:
                continue
            m = find_matches_for_item(prompt, update, prev_end_idx)
            if m is not None:
                candidates[update.item_idx] = m

        if not candidates:
            break

        # Prioritize earlier matches (processor.py:735).
        ordered = sorted(candidates.items(), key=lambda kv: kv[1])

        # REPLACE mode: only replace one non-empty item at a time (processor.py:738-750).
        chosen = ordered[0]
        item_idx, match = chosen

        out_chunks.append(prompt[prev_end_idx : match.start_idx])
        replacement = next(u.replacement for u in updates if u.item_idx == item_idx)
        out_chunks.append(replacement)
        resolved[item_idx] = match
        prev_end_idx = match.end_idx

    out_chunks.append(prompt[prev_end_idx:])
    final = [tok for chunk in out_chunks for tok in chunk]
    return final, resolved


# ---------------------------------------------------------------------------
# Synthetic data mirroring the REAL structure both sides produce:
#   transformers' Gemma4UnifiedProcessor.replace_video_token (verified at
#   transformers/models/gemma4_unified/processing_gemma4_unified.py:199-216):
#       "{ts1} {boi}{video_token * num_soft_tokens}{eoi} {ts2} {boi}{video_token * num_soft_tokens}{eoi} ..."
#   vLLM's Gemma4MultiModalProcessorInfo.get_video_repl (verified at
#   vllm/model_executor/models/gemma4_mm.py:398-433) builds the identical
#   per-frame pattern from `timestamps` + `num_soft_tokens_per_frame`.
# This is TRL's `_tokenize_prompts` output (grpo_trainer.py:1876-1894): the
# REAL processor's own tokenize=True call already performs this expansion
# client-side, before the prompt ever reaches vLLM.
# ---------------------------------------------------------------------------

VIDEO_TOKEN = 999
BOI_TOKEN = 900
EOI_TOKEN = 901
# Stand-ins for per-timestamp text tokens (e.g. "00:00 ", "00:01 ") -- real
# token IDs would differ per timestamp; using distinct small ints is enough
# to prove the structural point.
TS_TOKENS = [[10, 11], [12, 13], [14, 15]]  # 3 frames

NUM_SOFT_TOKENS_PER_FRAME = 4  # matches HF's single scalar per video (processing_gemma4_unified.py:200)

PREFIX = [1, 2, 3]  # e.g. "<bos><start_of_turn>user\n"
SUFFIX = [50, 51, 52]  # e.g. "Describe this video.<end_of_turn>"


def build_already_expanded_video_block() -> list[int]:
    block: list[int] = []
    for ts in TS_TOKENS:
        block += ts
        block.append(BOI_TOKEN)
        block += [VIDEO_TOKEN] * NUM_SOFT_TOKENS_PER_FRAME
        block.append(EOI_TOKEN)
    return block


def main():
    video_block = build_already_expanded_video_block()
    trl_prompt_ids = PREFIX + video_block + SUFFIX

    print(f"TRL's already-expanded prompt_ids sent to vLLM ({len(trl_prompt_ids)} tokens):")
    print(trl_prompt_ids)
    print()

    # vLLM's _get_prompt_updates (gemma4_mm.py:855-874): target is the SHORT,
    # single collapsed placeholder token; replacement is the FULL, correctly
    # -computed multi-frame expansion (derived from the real processor call
    # on the actual frames -- assumed here to be numerically identical to
    # TRL's own `video_block`, since Stage B already proved pixel/position-id
    # bit-identity). We use the *same* video_block as the "correct" replacement
    # to isolate the matching-mechanism bug from any pixel-value question.
    update = VideoUpdate(item_idx=0, target=[VIDEO_TOKEN], replacement=video_block)

    final_prompt, resolved = apply_matches(trl_prompt_ids, [update])

    print(f"vLLM's internally-reconstructed prompt_token_ids after _apply_prompt_updates ({len(final_prompt)} tokens):")
    print(final_prompt)
    print()

    print(f"Resolved match for item 0: {resolved[0]}")
    print()

    expected_len = len(trl_prompt_ids)  # if vLLM left it untouched, correctly recognizing it as already-expanded
    actual_len = len(final_prompt)

    n_orphaned_video_tokens = sum(1 for t in final_prompt if t == VIDEO_TOKEN) - sum(
        1 for t in video_block if t == VIDEO_TOKEN
    )

    print("=" * 70)
    if actual_len == expected_len and final_prompt == trl_prompt_ids:
        print("NO MISMATCH: vLLM's reconstruction is identical to TRL's original prompt.")
    else:
        print("MISMATCH CONFIRMED:")
        print(f"  TRL sent {expected_len} tokens; vLLM's internal prompt after matching is {actual_len} tokens")
        print(f"  ({actual_len - expected_len:+d} tokens vs. what TRL believes it sent)")
        print(
            f"  Orphaned `video_token` occurrences left in the tail, NOT part of the matched "
            f"replacement, hence NOT marked as `is_embed` multimodal positions: {n_orphaned_video_tokens}"
        )
        print()
        print(
            "  Effect: vLLM's rollout is generated conditioned on a DIFFERENT, corrupted-length "
            "context than what TRL's loss-time re-tokenization (which reuses the original, "
            "un-mangled prompt_ids -- confirmed via Stage C) assumes -- a large, structural "
            "context mismatch, not a subtle numerical one. Consistent with the observed "
            "35-117x sampling_logp_difference and near-zero importance_sampling_ratio."
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
