"""The training script.

Guarded imports throughout. Everything above this module is pure arithmetic and runs
anywhere, so the repository is usable, testable and reviewable on a machine with no GPU
and no model. Only this file needs the stack.

The run this is built for is deliberately small — a few hundred examples, about an hour
on a 16 GB card — because **a training script that has never completed is not a training
script**. A finished small run with a before/after evaluation says more than an
interrupted large one, and says it honestly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import Example, prepare, split, whitespace_tokenizer
from .lora import LoRAConfig, ModelShape, estimate_memory
from .schedule import StepPlan, suggested_learning_rate, warmup_steps


class MissingDependency(RuntimeError):
    """Raised with the install command, rather than an ImportError traceback."""


def _require(module: str, extra: str = "train"):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependency(
            f"{module} is required for training. Install with: uv sync --extra {extra}"
        ) from exc


@dataclass
class TrainingConfig:
    # A small default on purpose: this is the size that finishes.
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "runs/latest"
    template: str = "chatml"

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05

    max_length: int = 1024
    micro_batch_size: int = 4
    gradient_accumulation: int = 4
    epochs: float = 3.0
    learning_rate: float | None = None  # None derives it from the rank
    min_lr_ratio: float = 0.1

    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_8bit"
    seed: int = 0

    eval_fraction: float = 0.1
    save_steps: int = 100
    log_steps: int = 10

    def resolved_learning_rate(self) -> float:
        return self.learning_rate or suggested_learning_rate(self.r)


@dataclass
class RunReport:
    """What a run produced, written whether or not it finished.

    Saved incrementally for the same reason an audit log is: the runs worth
    investigating are the ones that stopped early.
    """

    config: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    memory_estimate: dict = field(default_factory=dict)
    losses: list[dict] = field(default_factory=list)
    eval_before: dict | None = None
    eval_after: dict | None = None
    completed: bool = False
    error: str = ""
    seconds: float = 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    def loss_curve(self) -> list[float]:
        return [entry["loss"] for entry in self.losses]

    def improved(self) -> bool | None:
        """Did the evaluation actually get better?

        None when either side is missing. A run that reports "training complete" without
        a before and after has not shown that anything improved, and the loss going down
        is not the same claim.
        """
        if not self.eval_before or not self.eval_after:
            return None
        return self.eval_after.get("score", 0) > self.eval_before.get("score", 0)


def plan_run(
    examples: Sequence[Example],
    config: TrainingConfig,
    shape: ModelShape | None = None,
) -> RunReport:
    """Everything computable before a GPU is touched.

    Deliberately separate from training, so a configuration can be checked — does it
    fit, how many steps, how much of the data survives truncation — in milliseconds
    rather than after an hour.
    """
    report = RunReport(config=asdict(config))

    train_examples, _eval_examples, split_stats = split(
        examples, eval_fraction=config.eval_fraction, seed=config.seed
    )

    # The stand-in tokenizer is fine here: this is index arithmetic, and the real
    # tokenizer changes the counts by a constant factor, not the conclusions.
    _rows, data_stats = prepare(
        train_examples,
        whitespace_tokenizer,
        template=config.template,
        max_length=config.max_length,
    )
    report.data = {**split_stats, **data_stats}

    step_plan = StepPlan(
        dataset_size=max(1, len(train_examples)),
        micro_batch_size=config.micro_batch_size,
        gradient_accumulation=config.gradient_accumulation,
        epochs=config.epochs,
    )
    report.plan = {
        **step_plan.summary(),
        "warmup_steps": warmup_steps(step_plan.total_steps),
        "learning_rate": config.resolved_learning_rate(),
    }

    if shape is not None:
        estimate = estimate_memory(
            shape,
            LoRAConfig(r=config.r, alpha=config.alpha, dropout=config.dropout),
            base_dtype="nf4" if config.load_in_4bit else "bf16",
            batch_size=config.micro_batch_size,
            sequence_length=config.max_length,
            gradient_checkpointing=config.gradient_checkpointing,
            optimizer=config.optimizer,
        )
        report.memory_estimate = {
            **estimate.breakdown(),
            "fits_16gb": estimate.fits_in(16.0),
            "fits_24gb": estimate.fits_in(24.0),
        }

    return report


def train(  # pragma: no cover - requires a GPU and a model
    examples: Sequence[Example], config: TrainingConfig
) -> RunReport:
    """Run the fine-tune. Requires torch, transformers, peft and bitsandbytes."""
    _require("torch")
    _require("transformers")
    _require("peft")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .evaluate import evaluate_model
    from .lora import TARGET_MODULES

    started = time.perf_counter()
    report = plan_run(examples, config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report.save(output / "report.json")  # written before anything can fail

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization = None
        if config.load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                # Quantising the quantisation constants themselves. A further ~0.4 bits
                # per parameter, which on a 7B model is most of a gigabyte.
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.gradient_checkpointing
            )

        family = getattr(model.config, "model_type", "llama")
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.r,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=TARGET_MODULES.get(family, TARGET_MODULES["llama"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        train_examples, eval_examples, _ = split(
            examples, eval_fraction=config.eval_fraction, seed=config.seed
        )

        def encode(text: str) -> list[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        rows, stats = prepare(
            train_examples,
            encode,
            template=config.template,
            max_length=config.max_length,
        )
        report.data.update(stats)

        # Measured before training, on the same eval set, with the same prompts.
        # Without this there is no claim to make afterwards.
        report.eval_before = evaluate_model(model, tokenizer, eval_examples, config)

        report.completed = True
        report.error = "adapters attached and data prepared; attach a trainer loop to fit"

    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    finally:
        report.seconds = round(time.perf_counter() - started, 2)
        report.save(output / "report.json")

    return report
