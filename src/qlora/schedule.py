"""Step arithmetic and learning-rate schedules.

Small, and worth getting right because two of these mistakes are common and both
produce a run that merely underperforms rather than one that fails.

**Effective batch size.** `batch_size` is what fits on the card; the number that
affects training is `batch_size × gradient_accumulation × devices`. A team that halves
the micro-batch to fit a longer sequence and forgets to double accumulation has halved
its effective batch and changed the optimisation problem without meaning to.

**Warmup on a short run.** The usual advice is 3% warmup. On a 200-step fine-tune that
is 6 steps, which is too few for Adam's moment estimates to settle — the first real
updates land with a badly conditioned optimiser. On short runs warmup should be a step
count, not a percentage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class ScheduleError(ValueError):
    pass


@dataclass
class StepPlan:
    dataset_size: int
    micro_batch_size: int
    gradient_accumulation: int
    epochs: float
    devices: int = 1

    def __post_init__(self) -> None:
        for name in ("dataset_size", "micro_batch_size", "gradient_accumulation", "devices"):
            if getattr(self, name) < 1:
                raise ScheduleError(f"{name} must be at least 1")
        if self.epochs <= 0:
            raise ScheduleError("epochs must be positive")

    @property
    def effective_batch_size(self) -> int:
        """The number that actually affects training."""
        return self.micro_batch_size * self.gradient_accumulation * self.devices

    @property
    def steps_per_epoch(self) -> int:
        # Floor, because a partial accumulation cycle does not produce an update.
        return max(1, self.dataset_size // self.effective_batch_size)

    @property
    def total_steps(self) -> int:
        return max(1, int(self.steps_per_epoch * self.epochs))

    @property
    def total_examples_seen(self) -> int:
        return int(self.dataset_size * self.epochs)

    def summary(self) -> dict:
        return {
            "effective_batch_size": self.effective_batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "examples_seen": self.total_examples_seen,
        }


def warmup_steps(total_steps: int, *, fraction: float = 0.03, minimum: int = 20) -> int:
    """Warmup as a fraction, with a floor, capped at a tenth of the run.

    The floor exists because 3% of a 200-step run is 6 steps, and Adam's moment
    estimates have not settled by then. The cap exists because the floor would
    otherwise consume most of a very short run.
    """
    if total_steps < 1:
        raise ScheduleError("total_steps must be at least 1")
    return max(1, min(max(int(total_steps * fraction), minimum), max(1, total_steps // 10)))


def linear_warmup_cosine_decay(
    step: int,
    *,
    total_steps: int,
    warmup: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """The standard schedule for fine-tuning: linear warmup, then cosine decay.

    Decay stops at `min_lr_ratio × peak`, not at zero. A learning rate that reaches
    exactly zero means the last steps do nothing, and on a short run those are the steps
    where the model is closest to where you want it.
    """
    if step < 0:
        raise ScheduleError("step must be non-negative")
    if warmup > total_steps:
        raise ScheduleError("warmup cannot exceed the total number of steps")

    if step < warmup:
        # From a nonzero value, so step 0 is a real update rather than a no-op.
        return peak_lr * (step + 1) / max(warmup, 1)

    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def constant_with_warmup(step: int, *, warmup: int, peak_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    return peak_lr


SCHEDULES = {
    "cosine": linear_warmup_cosine_decay,
    "constant": constant_with_warmup,
}


def suggested_learning_rate(r: int, *, base: float = 2e-4) -> float:
    """A starting learning rate for a given LoRA rank.

    LoRA tolerates rates one to two orders of magnitude above full fine-tuning, because
    only a small low-rank update is being learned. Higher ranks want slightly lower
    rates: more capacity means larger updates for the same gradient.

    A starting point, not a result. The right rate is found by running, and this exists
    so the first run is not wasted on an obviously wrong one.
    """
    if r < 1:
        raise ScheduleError("r must be at least 1")
    return round(base * (16 / r) ** 0.5, 8)
