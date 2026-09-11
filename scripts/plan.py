"""Check a fine-tuning configuration before spending an hour on it.

    python scripts/plan.py

Prints the VRAM breakdown, step plan and what the dataset loses to truncation. None of
this needs a GPU, and all of the mistakes it catches are cheaper to catch here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlora.data import Example  # noqa: E402
from qlora.lora import ModelShape, largest_config_that_fits  # noqa: E402
from qlora.train import TrainingConfig, plan_run  # noqa: E402

KNOWN = {
    "qwen2.5-0.5b": ModelShape(
        "Qwen2.5-0.5B",
        hidden_size=896,
        num_layers=24,
        num_attention_heads=14,
        intermediate_size=4864,
        vocab_size=151936,
        num_key_value_heads=2,
        family="qwen2",
        tie_word_embeddings=True,
    ),
    "qwen2.5-7b": ModelShape(
        "Qwen2.5-7B",
        hidden_size=3584,
        num_layers=28,
        num_attention_heads=28,
        intermediate_size=18944,
        vocab_size=152064,
        num_key_value_heads=4,
        family="qwen2",
    ),
    "llama3.1-8b": ModelShape(
        "Llama-3.1-8B",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        intermediate_size=14336,
        vocab_size=128256,
        num_key_value_heads=8,
        family="llama",
    ),
}

VRAM_GB = 16.0


def main() -> None:
    examples = [
        Example(instruction=f"Example instruction {i}", response=f"Response {i}")
        for i in range(500)
    ]

    for _name, shape in KNOWN.items():
        print(f"\n=== {shape.name} ({shape.total_parameters / 1e9:.2f}B params) ===")
        best = largest_config_that_fits(shape, VRAM_GB, sequence_length=1024)
        if "error" in best:
            print(f"  {best['error']}")
            continue

        print(
            f"  largest config fitting {VRAM_GB:.0f} GB: "
            f"r={best['r']}  batch={best['batch_size']}  "
            f"({best['estimated_gb']:.2f} GB, {best['headroom_gb']:.2f} GB spare)"
        )

        report = plan_run(
            examples,
            TrainingConfig(r=best["r"], alpha=best["alpha"], micro_batch_size=best["batch_size"]),
            shape=shape,
        )
        print("  plan: " + json.dumps(report.plan))
        print(
            "  data: "
            + json.dumps({k: report.data[k] for k in ("train", "eval", "supervised_fraction")})
        )


if __name__ == "__main__":
    main()
