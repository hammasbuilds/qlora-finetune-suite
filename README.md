# qlora-finetune-suite

[![ci](https://github.com/hammas159/qlora-finetune-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/hammas159/qlora-finetune-suite/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![core](https://img.shields.io/badge/core-no%20GPU%20required-success)
![license](https://img.shields.io/badge/license-MIT-green)

**The parts of fine-tuning that go wrong before the GPU is ever touched.**

VRAM budgeting · loss masking · leak-free splits · schedules · before/after evaluation.
The core is pure arithmetic and runs anywhere; only the training script needs the stack.

---

## Why the core has no dependencies

Every mistake that ruins a fine-tune is made **before** training starts, and every one of
them is checkable in milliseconds:

- a config that will OOM at step 900
- the loss computed on prompt tokens as well as the response
- truncation that silently removed the answer from a fifth of the dataset
- an eval set sharing examples with the training set
- an effective batch size half what you think it is

So those live in code that needs no GPU, no model and no download — and they are tested
exhaustively. `make plan` tells you whether a run will fit and how many steps it will
take, before you spend the hour.

## 1. VRAM budgeting

```
Qwen2.5-7B (7.62B params) on a 16 GB card, 1024 tokens

QLoRA nf4       5.52 GB   fits        optimizer   0.30 GB
LoRA  bf16     16.16 GB   does not    optimizer   0.30 GB
full FT bf16   86.64 GB   does not    optimizer  56.74 GB   ← the real cost
```

People budget for weights and are surprised. **In full fine-tuning the optimiser is the
larger cost** — Adam keeps two fp32 moments per trainable parameter, so a 7B model needs
~56 GB of optimiser state before a single activation. QLoRA removes almost all of it,
because only the adapters are trainable.

```python
largest_config_that_fits(QWEN_7B, vram_gb=16.0, sequence_length=1024)
# {"r": 64, "alpha": 128, "batch_size": 8, "estimated_gb": 11.95, "headroom_gb": 4.05}
```

Parameter counts account for **grouped-query attention** (k and v are smaller than q) and
**tied embeddings** (small models share one matrix between input and output). Ignoring
the second overstates a 0.5B model by about a quarter, because on a small model with a
150k vocabulary the embedding *is* a large fraction of the total. Counts land within 1%
of published figures: 0.49B, 7.62B, 8.03B.

## 2. Loss masking — the bug that silently ruins instruction tuning

A training example is a prompt and a response. The loss must be computed on the
**response only**. Include the prompt and the model spends most of its gradient learning
to generate *questions*, because in a typical instruction dataset the prompt is longer
than the answer.

**The run completes. The loss curve looks fine** — it goes down, because predicting
prompts is a real task the model gets better at. The result is a model slightly worse at
what you wanted and noticeably better at writing instructions. Nothing in the logs says
so.

```python
def test_prompt_tokens_are_masked_out_of_the_loss():
    row = tokenise(Example("What is 2+2?", "Four."), tokenizer)
    assert all(label == IGNORE_INDEX for label in row.labels[:row.prompt_length])
```

The split is **structural**, not a text search. Rendering one string and locating the
response inside it breaks whenever the response also appears in the prompt — which is
constant in summarisation, extraction and translation data. `render()` returns the
prompt and completion separately because that boundary *is* the mask.

### Truncation can delete the answer

Cutting a long example at the sequence limit can remove the entire response, leaving a
row with **no supervised tokens**. It contributes nothing, it does not error, and it
quietly shrinks the dataset. So rows are checked, `truncate="left"` preserves the
response, and `prepare()` reports what was lost:

```python
{"kept": 412, "dropped_no_response": 88, "truncated": 96,
 "supervised_fraction": 0.29}
```

`supervised_fraction` is the share of compute that trains what you care about. Worth
knowing before a long run.

## 3. Leak-free splits

Deduplication happens **before** the split. Deduplicating each side separately leaves
cross-split duplicates untouched — which is exactly the leak, and it turns the eval into
a memorisation check that always passes.

```python
train, evaluation, stats = split(examples)
assert stats["cross_split_duplicates"] == 0   # asserted, not assumed
```

## 4. Schedules

**Effective batch size** is `micro_batch × accumulation × devices`. Halving the
micro-batch to fit a longer sequence and forgetting to double accumulation halves the
effective batch and changes the optimisation problem without meaning to.

**Warmup has a floor.** The usual 3% advice gives 6 steps on a 200-step fine-tune, which
is too few for Adam's moment estimates to settle. On short runs warmup should be a step
count.

**Decay stops above zero.** A learning rate reaching exactly zero means the last steps do
nothing — and on a short run those are the steps closest to where you want the model.

## 5. Before/after evaluation

The claim a fine-tuning project has to support is *"this model got better at the thing I
trained it for"*. **The loss curve does not support it** — loss falls whenever the model
gets better at predicting the training distribution, including when it is memorising, and
including when the masking was wrong.

```python
compare(before_predictions, after_predictions, references).summary()
# {"before": 0.50, "after": 0.90, "delta": 0.40, "improved": True,
#  "regressed_examples": 1, "regression_rate": 0.5}
```

**Regressions are surfaced, not averaged away.** A mean improvement can hide a model that
got much better at most of the set and broke on a subset — and the broken subset is
frequently the one that mattered.

Perplexity is deliberately excluded. It is the easiest number to produce and among the
least informative: it improves when the model learns the dataset's formatting quirks,
which is not a capability anyone wanted.

## Usage

```bash
make install        # core only, no GPU stack
make test           # 75 tests, no GPU, no model, no download
make plan           # will this config fit? how many steps?

make install-train  # torch, transformers, peft, bitsandbytes
```

```python
report = plan_run(examples, TrainingConfig(r=16), shape=QWEN_7B)
report.plan             # steps, warmup, learning rate
report.memory_estimate  # breakdown + fits_16gb
report.data             # split sizes, truncation losses, supervised fraction
```

## On the training run itself

The default base model is **Qwen2.5-0.5B-Instruct**, deliberately. A training script that
has never completed is not a training script, and a finished small run with a real
before/after comparison says more — and says it honestly — than an interrupted large one.

`RunReport` is written to disk *before* anything can fail and again at the end, for the
same reason an audit log is: the runs worth investigating are the ones that stopped early.

**`report.improved()` returns `None` when either evaluation is missing.** A run that
reports "training complete" without a before and an after has not shown that anything
improved.

## Tests

**75 tests. No GPU, no model, no download.**

| Covered | |
|---|---|
| Model shapes | parameter counts, grouped-query attention, **tied embeddings**, invalid configs |
| LoRA | α/r scaling invariance, rank linearity, attention-only trade, embedding cost |
| Memory | QLoRA fits / full FT does not, **optimiser dominates**, 4-bit ratio, checkpointing, 8-bit Adam, sequence length, config search |
| **Masking** | prompt masked, response supervised, alignment, off-by-one, structural split, truncation destroying or preserving the answer |
| Templates | ChatML / Alpaca / Llama-3, system and input fields |
| Splitting | **no cross-split duplicates**, dedup before split, whitespace-insensitive, reproducible |
| Packing | batching, limits, boundaries recorded, masking survives |
| Scheduling | effective batch size, step counts, warmup floor and cap, warmup→decay shape, nonzero floor |
| Evaluation | exact/contains/F1, mismatched inputs, delta, **regressions surfaced** |

## Limits

- **The trainer loop itself is not included.** The suite owns everything around it —
  which is where the mistakes are — and hands off to HF `Trainer`, TRL or a hand-written
  loop. That is a real gap and it is stated rather than hidden.
- Memory figures are **estimates**. Fragmentation and kernel workspaces are not modelled,
  and the estimate is deliberately a little pessimistic: a config predicted to fit and
  then failing at step 900 wastes an hour; one predicted not to fit costs nothing to
  re-check.
- The stand-in tokenizer is not a BPE. It exists so masking and truncation arithmetic can
  be verified exactly; a real tokenizer changes the counts by a constant factor, not the
  conclusions.
- Scorers are exact-match, containment and token-F1 — crude, deterministic, and honest
  about what they measure. An LLM judge fits the same signature.
- **No run has been completed on real data.** The arithmetic is verified; the fine-tune
  is not.

## License

MIT

---

## Run it yourself

```bash
git clone https://github.com/hammas159/qlora-finetune-suite
cd qlora-finetune-suite

uv sync --group dev      # or: pip install -e ".[dev]"
make test                # 75 tests, no GPU, no model, no download
make plan                # will my config fit? how many steps?
```

`make plan` is the point — it answers before you spend the hour:

```
=== Qwen2.5-7B (7.62B params) ===
  largest config fitting 16 GB: r=64  batch=8  (11.95 GB, 4.05 GB spare)
  plan: {"effective_batch_size": 32, "total_steps": 42, "warmup_steps": 4, ...}
  data: {"train": 450, "eval": 50, "supervised_fraction": 0.2857}
```

To actually train, add the stack:

```bash
make install-train       # torch, transformers, peft, bitsandbytes
```

## Problems hit while building this

**Parameter counts were 25% high on small models.** Counting the input embedding and the
output head separately overstates any model that *ties* them — and small models almost
always do. On a 0.5B model with a 150k vocabulary the embedding is a large share of the
total, so `Qwen2.5-0.5B` came out at 0.63B instead of 0.49B, and every VRAM figure
derived from it was wrong in the same direction. *Fixed* with a `tie_word_embeddings`
flag; counts now land within 1% of published figures (0.49B, 7.62B, 8.03B).

**Two failures were my tests, not the code, and it is worth being precise about which.**
One fixture asserted a fine-tune "improved" on data where the mean had actually got
worse — I had wanted a case where the average rises while one example regresses, and
built the opposite. The other was a shim in the offline test runner that ignored
`pytest.approx`'s relative tolerance, so an exact-to-three-decimals comparison failed on
the fourth. Both are recorded because *"the test was wrong"* and *"the code was wrong"*
are different claims and conflating them erodes trust in the suite.

**No fine-tune has actually been run.** The arithmetic is verified to the published
figures; the training loop is not included and no model has been trained with this. That
is stated here rather than left for someone to discover.
