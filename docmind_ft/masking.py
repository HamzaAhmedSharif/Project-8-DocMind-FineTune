"""Completion-only loss-mask check (gate I3, script 3.11).

Pure helper: given the token ids of the prompt part, the full input ids
and the labels of one training example, verify that every prompt token is
masked (label -100) and at least ``min_completion_frac`` of the completion
tokens carry a label. This is the test that detects a TRL option rename:
it checks the labels the trainer actually produced, not the option names.
"""

from __future__ import annotations


def check_completion_mask(
    prompt_ids: list[int],
    input_ids: list[int],
    labels: list[int],
    min_completion_frac: float = 0.95,
) -> dict:
    """Check one example's loss mask. Returns a dict with ``ok`` and detail."""
    if len(input_ids) != len(labels):
        return {
            "ok": False,
            "reason": f"input_ids length {len(input_ids)} != labels length {len(labels)}",
        }
    if not prompt_ids:
        return {"ok": False, "reason": "empty prompt_ids"}
    # Locate the prompt span: either at offset 0, or at offset 1 when the
    # tokenizer prepends a BOS token the prompt tokenization did not have.
    offset = None
    if input_ids[: len(prompt_ids)] == prompt_ids:
        offset = 0
    elif input_ids[1 : 1 + len(prompt_ids)] == prompt_ids:
        offset = 1
    if offset is None:
        return {
            "ok": False,
            "reason": "prompt_ids are not a prefix of input_ids (offset 0 or 1)",
        }
    prompt_end = offset + len(prompt_ids)
    prompt_labels = labels[offset:prompt_end]
    completion_labels = labels[prompt_end:]
    prompt_masked = sum(1 for t in prompt_labels if t == -100)
    completion_trained = sum(1 for t in completion_labels if t != -100)
    completion_total = len(completion_labels)
    frac = completion_trained / completion_total if completion_total else 0.0
    ok = prompt_masked == len(prompt_labels) and frac >= min_completion_frac
    reason = ""
    if prompt_masked != len(prompt_labels):
        reason = f"{len(prompt_labels) - prompt_masked} prompt token(s) unmasked"
    elif frac < min_completion_frac:
        reason = f"completion trained fraction {frac:.3f} < {min_completion_frac}"
    return {
        "ok": ok,
        "reason": reason,
        "offset": offset,
        "prompt_len": len(prompt_labels),
        "prompt_masked": prompt_masked,
        "completion_len": completion_total,
        "completion_trained": completion_trained,
        "completion_trained_frac": frac,
    }
