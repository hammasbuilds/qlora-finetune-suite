"""Before/after evaluation.

The claim a fine-tuning project has to support is *"this model got better at the thing
I trained it for"*. The loss curve going down does not support it — loss falls whenever
the model gets better at predicting the training distribution, including when it is
memorising, and including when the masking was wrong and it is learning to generate
prompts.

So evaluation is:

  **the same held-out set** before and after
  **the same prompts**, decoded the same way
  **a metric chosen for the task**, not perplexity

Perplexity is excluded deliberately. It is the easiest number to produce and among the
least informative: it improves when the model learns the dataset's formatting quirks,
which is not the capability anyone wanted.

The scorers here are exact-match, containment and token-F1 — crude, but computable
without a judge model, deterministic, and honest about what they measure. An LLM judge
fits behind the same `Scorer` signature.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .data import Example, render

# (prediction, reference) -> score in [0, 1]
Scorer = Callable[[str, str], float]

_WORD = re.compile(r"[\w']+")


def _normalise(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(_normalise(prediction)) == " ".join(_normalise(reference)))


def contains(prediction: str, reference: str) -> float:
    """Does the prediction contain the reference answer?

    The right metric for extraction and short-answer tasks, where a model that answers
    correctly and then adds a sentence of explanation should not score zero.
    """
    reference_text = " ".join(_normalise(reference))
    return float(reference_text and reference_text in " ".join(_normalise(prediction)))


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall.

    Partial credit, which matters on generative tasks where exact match is almost
    always zero and therefore tells you nothing about whether the model improved.
    """
    predicted = Counter(_normalise(prediction))
    expected = Counter(_normalise(reference))
    if not predicted or not expected:
        return float(predicted == expected)

    overlap = sum((predicted & expected).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return round(2 * precision * recall / (precision + recall), 6)


SCORERS: dict[str, Scorer] = {
    "exact_match": exact_match,
    "contains": contains,
    "token_f1": token_f1,
}


@dataclass
class EvalResult:
    scorer: str
    n: int
    score: float
    per_example: list[float] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"scorer": self.scorer, "n": self.n, "score": self.score}


def evaluate(
    predictions: Sequence[str], references: Sequence[str], *, scorer: str = "token_f1"
) -> EvalResult:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}; have {sorted(SCORERS)}")

    fn = SCORERS[scorer]
    scores = [fn(p, r) for p, r in zip(predictions, references, strict=False)]
    return EvalResult(
        scorer=scorer,
        n=len(scores),
        score=round(sum(scores) / len(scores), 6) if scores else 0.0,
        per_example=scores,
        predictions=list(predictions),
    )


@dataclass
class Comparison:
    before: EvalResult
    after: EvalResult

    @property
    def delta(self) -> float:
        return round(self.after.score - self.before.score, 6)

    @property
    def improved(self) -> bool:
        return self.delta > 0

    def regressions(self) -> list[int]:
        """Examples the fine-tune made worse.

        Reported because a mean improvement can hide a model that got much better at
        most of the set and broke on a subset — and the broken subset is frequently the
        one that mattered.
        """
        return [
            i
            for i, (b, a) in enumerate(
                zip(self.before.per_example, self.after.per_example, strict=False)
            )
            if a < b
        ]

    def summary(self) -> dict:
        regressed = self.regressions()
        return {
            "scorer": self.before.scorer,
            "n": self.before.n,
            "before": self.before.score,
            "after": self.after.score,
            "delta": self.delta,
            "improved": self.improved,
            "regressed_examples": len(regressed),
            # The honest headline. A model better on average and worse on a fifth of the
            # set is a different result from one better everywhere.
            "regression_rate": (round(len(regressed) / self.before.n, 4) if self.before.n else 0.0),
        }


def compare(
    before_predictions: Sequence[str],
    after_predictions: Sequence[str],
    references: Sequence[str],
    *,
    scorer: str = "token_f1",
) -> Comparison:
    return Comparison(
        before=evaluate(before_predictions, references, scorer=scorer),
        after=evaluate(after_predictions, references, scorer=scorer),
    )


def evaluate_model(  # pragma: no cover - requires a model
    model,
    tokenizer,
    examples: Sequence[Example],
    config,
    *,
    scorer: str = "token_f1",
    max_new_tokens: int = 128,
) -> dict:
    """Generate on the eval set and score it.

    Greedy decoding, deliberately: sampling makes the before/after difference partly a
    difference in random draws, and the comparison stops meaning anything.
    """
    import torch

    predictions: list[str] = []
    references = [e.response for e in examples]

    model.eval()
    for example in examples:
        prompt, _ = render(example, config.template)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predictions.append(text)

    return evaluate(predictions, references, scorer=scorer).to_dict()
