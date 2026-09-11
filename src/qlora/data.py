"""Dataset preparation, and the masking that decides whether fine-tuning works at all.

**The bug that silently ruins instruction tuning.**

A training example is a prompt and a response. The model must learn to produce the
response *given* the prompt — so the loss has to be computed on the response tokens
only. If the prompt tokens are included, the model spends most of its gradient learning
to generate **questions**, because in a typical instruction dataset the prompt is longer
than the answer.

The run completes. The loss curve looks fine — it goes down, because predicting prompts
is a real task the model gets better at. The result is a model that has become slightly
worse at the thing you wanted and noticeably better at producing instructions. Nothing
in the training logs says so.

`-100` is the convention: PyTorch's cross-entropy ignores that label index. So masking
is setting label positions to `-100`, and the property worth asserting is simple —
**every masked position is prompt, every unmasked position is response**.

The second trap is **truncation**. Cutting a long example at the sequence limit can
remove the entire response, leaving a training row with no supervised tokens at all. It
contributes nothing, it does not error, and it quietly shrinks the dataset.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

IGNORE_INDEX = -100

# A tokenizer is injected so the whole module is testable without downloading one.
# text -> token ids
Tokenizer = Callable[[str], list[int]]


def whitespace_tokenizer(text: str) -> list[int]:
    """A stand-in tokenizer: one stable id per word.

    Not a real BPE, and not pretending to be. It exists so masking, truncation and
    packing can be verified exactly, because those are index arithmetic and index
    arithmetic does not care which tokenizer produced the indices.
    """
    return [
        int(hashlib.blake2b(w.encode(), digest_size=4).hexdigest(), 16) % 30000
        for w in text.split()
    ]


@dataclass
class Example:
    instruction: str
    response: str
    system: str = ""
    input: str = ""

    @property
    def fingerprint(self) -> str:
        """Identity for deduplication, insensitive to whitespace and case."""
        joined = " ".join(
            re.sub(r"\s+", " ", part.strip().lower())
            for part in (self.system, self.instruction, self.input, self.response)
        )
        return hashlib.blake2b(joined.encode(), digest_size=16).hexdigest()


# --- chat templates ---------------------------------------------------------------

TEMPLATES: dict[str, dict[str, str]] = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "assistant_suffix": "<|im_end|>\n",
    },
    "alpaca": {
        "system": "{content}\n\n",
        "user": "### Instruction:\n{content}\n\n",
        "assistant_prefix": "### Response:\n",
        "assistant_suffix": "\n",
    },
    "llama3": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_suffix": "<|eot_id|>",
    },
}


def render(example: Example, template: str = "chatml") -> tuple[str, str]:
    """Return (prompt, completion).

    Split rather than concatenated, because the split *is* the masking boundary.
    Rendering one string and searching for the response afterwards fails whenever the
    response text also appears in the prompt — which happens constantly in
    summarisation, extraction and translation data.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; have {sorted(TEMPLATES)}")
    parts = TEMPLATES[template]

    prompt = ""
    if example.system:
        prompt += parts["system"].format(content=example.system)

    user = example.instruction
    if example.input:
        user += f"\n\n{example.input}"
    prompt += parts["user"].format(content=user)
    prompt += parts["assistant_prefix"]

    completion = example.response + parts["assistant_suffix"]
    return prompt, completion


@dataclass
class TokenisedExample:
    input_ids: list[int]
    labels: list[int]
    prompt_length: int

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)

    @property
    def is_usable(self) -> bool:
        """A row with no supervised tokens contributes nothing and is not an error.

        Truncation produces these silently, which is why this is checked rather than
        assumed.
        """
        return self.supervised_tokens > 0


def tokenise(
    example: Example,
    tokenizer: Tokenizer,
    *,
    template: str = "chatml",
    max_length: int = 1024,
    truncate: str = "right",
) -> TokenisedExample:
    """Tokenise one example and mask the prompt out of the loss."""
    prompt, completion = render(example, template)
    prompt_ids = tokenizer(prompt)
    completion_ids = tokenizer(completion)

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)

    if len(input_ids) > max_length:
        if truncate == "left":
            # Drop from the *front*, which preserves the response. For a long-context
            # example that is almost always the right trade: losing the start of the
            # prompt costs some context, losing the response costs the entire example.
            overflow = len(input_ids) - max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
        else:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

    return TokenisedExample(input_ids=input_ids, labels=labels, prompt_length=len(prompt_ids))


def prepare(
    examples: Sequence[Example],
    tokenizer: Tokenizer,
    *,
    template: str = "chatml",
    max_length: int = 1024,
    truncate: str = "right",
    drop_unusable: bool = True,
) -> tuple[list[TokenisedExample], dict]:
    """Tokenise a dataset, and report what was lost.

    The report matters. Silently dropping a fifth of a dataset to truncation is the
    difference between a run that underperforms for a reason you can find and one that
    underperforms mysteriously.
    """
    rows = [
        tokenise(e, tokenizer, template=template, max_length=max_length, truncate=truncate)
        for e in examples
    ]
    unusable = [r for r in rows if not r.is_usable]
    kept = [r for r in rows if r.is_usable] if drop_unusable else rows

    supervised = sum(r.supervised_tokens for r in kept)
    total = sum(len(r.input_ids) for r in kept)

    return kept, {
        "examples": len(examples),
        "kept": len(kept),
        "dropped_no_response": len(unusable),
        "truncated": sum(1 for r in rows if len(r.input_ids) >= max_length),
        "total_tokens": total,
        "supervised_tokens": supervised,
        # The fraction of compute that trains the thing you care about. Low here means
        # long prompts and short answers, which is worth knowing before a long run.
        "supervised_fraction": round(supervised / total, 4) if total else 0.0,
    }


# --- splitting ---------------------------------------------------------------------


def deduplicate(examples: Sequence[Example]) -> tuple[list[Example], int]:
    """Remove exact duplicates, whitespace- and case-insensitively.

    Instruction datasets assembled from several sources are full of them, and a
    duplicate that lands in both train and eval turns the eval into a memorisation
    check that always passes.
    """
    seen: set[str] = set()
    unique: list[Example] = []
    for example in examples:
        fingerprint = example.fingerprint
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(example)
    return unique, len(examples) - len(unique)


def split(
    examples: Sequence[Example],
    *,
    eval_fraction: float = 0.1,
    seed: int = 0,
    deduplicate_first: bool = True,
) -> tuple[list[Example], list[Example], dict]:
    """Train/eval split with no example on both sides.

    Deduplication happens *before* the split, not after. Deduplicating each side
    separately leaves cross-split duplicates untouched, which is exactly the leak.
    """
    working = list(examples)
    removed = 0
    if deduplicate_first:
        working, removed = deduplicate(working)

    rng = random.Random(seed)
    rng.shuffle(working)

    eval_size = max(1, int(len(working) * eval_fraction)) if working else 0
    evaluation, training = working[:eval_size], working[eval_size:]

    train_fingerprints = {e.fingerprint for e in training}
    overlap = sum(1 for e in evaluation if e.fingerprint in train_fingerprints)

    return (
        training,
        evaluation,
        {
            "train": len(training),
            "eval": len(evaluation),
            "duplicates_removed": removed,
            # Must be zero. Asserted in the tests rather than trusted.
            "cross_split_duplicates": overlap,
        },
    )


# --- packing -----------------------------------------------------------------------


@dataclass
class PackedBatch:
    input_ids: list[int]
    labels: list[int]
    boundaries: list[int] = field(default_factory=list)


def pack(rows: Sequence[TokenisedExample], *, max_length: int = 1024) -> list[PackedBatch]:
    """Concatenate short examples up to the sequence limit.

    On a dataset of short examples, padding wastes most of the compute: a batch padded
    to 1024 where the average example is 180 tokens spends 80% of its FLOPs on padding.
    Packing removes that.

    Boundaries are recorded so an attention mask can prevent one example attending to
    the next. Without that mask, packing teaches the model that unrelated examples
    follow one another — cheaper *and* worse, which is the failure mode to avoid.
    """
    batches: list[PackedBatch] = []
    current = PackedBatch(input_ids=[], labels=[])

    for row in rows:
        if len(row.input_ids) > max_length:
            continue  # cannot be packed; it was already truncated upstream
        if len(current.input_ids) + len(row.input_ids) > max_length:
            if current.input_ids:
                batches.append(current)
            current = PackedBatch(input_ids=[], labels=[])
        current.boundaries.append(len(current.input_ids))
        current.input_ids.extend(row.input_ids)
        current.labels.extend(row.labels)

    if current.input_ids:
        batches.append(current)
    return batches
