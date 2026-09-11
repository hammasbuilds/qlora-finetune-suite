"""LoRA arithmetic and VRAM budgeting.

The whole point of QLoRA is fitting a model you could not otherwise train onto the card
you actually have. That makes the arithmetic the design, not a detail — and it is
arithmetic, so it can be checked before spending an hour discovering an out-of-memory
error at step 900.

**LoRA.** A weight update ΔW of shape (d_out, d_in) is approximated by BA, where B is
(d_out, r) and A is (r, d_in). Instead of d_out·d_in trainable parameters you have
r·(d_out + d_in), which for r=16 on a 4096×4096 projection is 131k instead of 16.7M —
**0.8%**.

    W' = W + (α/r)·BA

The α/r scaling exists so that changing r does not change the effective learning rate.
Raising r without it silently makes the update larger, and the run that "worked at r=8
but diverged at r=64" is almost always this.

**Where the memory actually goes.** People budget for weights and are surprised. In full
fine-tuning the optimiser is the larger cost: Adam keeps two fp32 moments per trainable
parameter, so a 7B model needs ~56 GB of optimiser state alone before any activation.
QLoRA removes almost all of it, because only the adapters are trainable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Bytes per parameter by storage format.
DTYPE_BYTES = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "nf4": 0.5, "fp4": 0.5,
}

# Attention and MLP projections, by architecture family. Naming differs; the targets
# do not.
TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}

# Attention-only is the common shortcut and is usually the wrong trade: the MLP holds
# roughly two thirds of the parameters, and excluding it costs more quality than the
# memory it saves.
ATTENTION_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]


class ConfigError(ValueError):
    pass


@dataclass
class ModelShape:
    """Enough of an architecture to compute its cost."""

    name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    num_key_value_heads: int | None = None   # None means multi-head, not grouped-query
    family: str = "llama"
    # Small models usually share one matrix between the input embedding and the output
    # head. Counting it twice overstates a 0.5B model by around 25%, because on a small
    # model with a 150k vocabulary the embedding *is* a large fraction of the total.
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ConfigError("hidden_size must divide evenly by the head count")
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def total_parameters(self) -> int:
        """Approximate parameter count, accounting for grouped-query attention.

        GQA shares key and value projections across query heads, so k_proj and v_proj
        are smaller than q_proj. Treating them as equal - as most back-of-envelope
        counts do - overestimates a modern model by several percent.
        """
        kv_size = self.num_key_value_heads * self.head_dim

        attention = (
            self.hidden_size * self.hidden_size        # q
            + self.hidden_size * kv_size               # k
            + self.hidden_size * kv_size               # v
            + self.hidden_size * self.hidden_size      # o
        )
        mlp = 3 * self.hidden_size * self.intermediate_size   # gate, up, down
        per_layer = attention + mlp + 2 * self.hidden_size     # two RMSNorms

        embeddings = self.vocab_size * self.hidden_size
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size

        return self.num_layers * per_layer + embeddings + lm_head + self.hidden_size


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES["llama"]))
    # Training the embedding and output layers as well. Rarely worth it, and the reason
    # is concrete: they are the largest matrices in a small model, so the memory cost is
    # high and the quality gain is usually small unless the vocabulary is changing.
    train_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ConfigError("r must be at least 1")
        if self.alpha <= 0:
            raise ConfigError("alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("dropout must be in [0, 1)")
        if not self.target_modules:
            raise ConfigError("no target modules; the adapter would train nothing")

    @property
    def scaling(self) -> float:
        """α/r. Kept constant when r changes, or the effective learning rate moves."""
        return self.alpha / self.r


def adapter_parameters(shape: ModelShape, config: LoRAConfig) -> dict:
    """Trainable parameters introduced by the adapters, per module and in total."""
    kv_size = shape.num_key_value_heads * shape.head_dim

    dimensions = {
        "q_proj": (shape.hidden_size, shape.hidden_size),
        "k_proj": (kv_size, shape.hidden_size),
        "v_proj": (kv_size, shape.hidden_size),
        "o_proj": (shape.hidden_size, shape.hidden_size),
        "gate_proj": (shape.intermediate_size, shape.hidden_size),
        "up_proj": (shape.intermediate_size, shape.hidden_size),
        "down_proj": (shape.hidden_size, shape.intermediate_size),
        "query_key_value": (3 * shape.hidden_size, shape.hidden_size),
        "dense": (shape.hidden_size, shape.hidden_size),
        "dense_h_to_4h": (shape.intermediate_size, shape.hidden_size),
        "dense_4h_to_h": (shape.hidden_size, shape.intermediate_size),
    }

    per_module: dict[str, int] = {}
    for module in config.target_modules:
        if module not in dimensions:
            raise ConfigError(f"unknown target module {module!r}")
        d_out, d_in = dimensions[module]
        # B is (d_out, r) and A is (r, d_in).
        per_module[module] = config.r * (d_out + d_in) * shape.num_layers

    total = sum(per_module.values())
    if config.train_embeddings:
        total += 2 * shape.vocab_size * shape.hidden_size

    base = shape.total_parameters
    return {
        "per_module": per_module,
        "trainable": total,
        "base": base,
        "trainable_fraction": round(total / base, 8),
        "trainable_percent": round(100 * total / base, 4),
    }


@dataclass
class MemoryEstimate:
    base_weights_gb: float
    adapter_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return round(
            self.base_weights_gb + self.adapter_gb + self.gradients_gb
            + self.optimizer_gb + self.activations_gb + self.overhead_gb,
            3,
        )

    def fits_in(self, vram_gb: float) -> bool:
        return self.total_gb <= vram_gb

    def breakdown(self) -> dict:
        return {
            "base_weights": self.base_weights_gb,
            "adapters": self.adapter_gb,
            "gradients": self.gradients_gb,
            "optimizer": self.optimizer_gb,
            "activations": self.activations_gb,
            "overhead": self.overhead_gb,
            "total": self.total_gb,
        }


def estimate_memory(
    shape: ModelShape,
    config: LoRAConfig | None,
    *,
    base_dtype: str = "nf4",
    compute_dtype: str = "bf16",
    batch_size: int = 1,
    sequence_length: int = 1024,
    gradient_checkpointing: bool = True,
    optimizer: str = "adamw",
) -> MemoryEstimate:
    """Estimate peak VRAM. `config=None` means full fine-tuning.

    An estimate, not a guarantee - fragmentation and kernel workspaces are real and not
    modelled. It is deliberately a little pessimistic, because a config predicted to fit
    and then failing at step 900 wastes an hour, while one predicted not to fit costs
    nothing to re-check.
    """
    if base_dtype not in DTYPE_BYTES:
        raise ConfigError(f"unknown dtype {base_dtype!r}")

    giga = 1024 ** 3
    base_parameters = shape.total_parameters

    base_weights = base_parameters * DTYPE_BYTES[base_dtype] / giga

    if config is None:
        trainable = base_parameters
        adapter = 0.0
    else:
        trainable = adapter_parameters(shape, config)["trainable"]
        adapter = trainable * DTYPE_BYTES[compute_dtype] / giga

    # Gradients exist only for trainable parameters - the saving QLoRA is built on.
    gradients = trainable * DTYPE_BYTES[compute_dtype] / giga

    # Adam keeps two fp32 moments per trainable parameter. In full fine-tuning this is
    # the dominant cost and the reason a 7B model will not train on a 24 GB card.
    states = {"adamw": 2, "adamw_8bit": 0.5, "sgd": 0, "sgd_momentum": 1}
    if optimizer not in states:
        raise ConfigError(f"unknown optimizer {optimizer!r}")
    optimizer_gb = trainable * states[optimizer] * 4.0 / giga

    # Activations scale with batch, sequence and depth. Gradient checkpointing trades
    # roughly a third more compute for storing only layer boundaries.
    per_token = shape.hidden_size * DTYPE_BYTES[compute_dtype]
    if gradient_checkpointing:
        activations = batch_size * sequence_length * per_token * shape.num_layers**0.5 * 4 / giga
    else:
        activations = batch_size * sequence_length * per_token * shape.num_layers * 12 / giga

    # Logits are frequently the surprise on a large vocabulary: batch × seq × vocab in
    # fp32 for the loss, which at 150k vocab dwarfs the model on a long sequence.
    logits = batch_size * sequence_length * shape.vocab_size * 4 / giga

    return MemoryEstimate(
        base_weights_gb=round(base_weights, 3),
        adapter_gb=round(adapter, 4),
        gradients_gb=round(gradients, 4),
        optimizer_gb=round(optimizer_gb, 4),
        activations_gb=round(activations + logits, 3),
        overhead_gb=0.8,   # CUDA context, kernels, fragmentation
    )


def largest_config_that_fits(
    shape: ModelShape, vram_gb: float, *, sequence_length: int = 1024, **kw
) -> dict:
    """Search rank and batch size for the biggest configuration that fits.

    Ordered so the answer is the most capable option rather than the first one found:
    rank matters more than batch size for adaptation quality, and batch size can be
    recovered with gradient accumulation at no memory cost.
    """
    best: dict | None = None
    for r in (64, 32, 16, 8, 4):
        for batch_size in (8, 4, 2, 1):
            config = LoRAConfig(r=r, alpha=2 * r)
            estimate = estimate_memory(
                shape, config, batch_size=batch_size,
                sequence_length=sequence_length, **kw
            )
            if estimate.fits_in(vram_gb):
                candidate = {
                    "r": r, "alpha": 2 * r, "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "estimated_gb": estimate.total_gb,
                    "headroom_gb": round(vram_gb - estimate.total_gb, 3),
                    "breakdown": estimate.breakdown(),
                }
                if best is None or (r, batch_size) > (best["r"], best["batch_size"]):
                    best = candidate
    return best or {"error": f"nothing fits in {vram_gb} GB; reduce sequence_length"}
