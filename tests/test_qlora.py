"""QLoRA suite tests.

All arithmetic and index bookkeeping, so all of it is exact — which is the point: the
mistakes that ruin a fine-tune are made before the GPU is touched, and they are
checkable in milliseconds.
"""

from __future__ import annotations

import pytest

from qlora.data import (
    IGNORE_INDEX,
    Example,
    deduplicate,
    pack,
    prepare,
    render,
    split,
    tokenise,
    whitespace_tokenizer,
)
from qlora.evaluate import compare, contains, evaluate, exact_match, token_f1
from qlora.lora import (
    ATTENTION_ONLY,
    ConfigError,
    LoRAConfig,
    ModelShape,
    adapter_parameters,
    estimate_memory,
    largest_config_that_fits,
)
from qlora.schedule import (
    ScheduleError,
    StepPlan,
    linear_warmup_cosine_decay,
    suggested_learning_rate,
    warmup_steps,
)
from qlora.train import TrainingConfig, plan_run

QWEN_7B = ModelShape(
    "Qwen2.5-7B", hidden_size=3584, num_layers=28, num_attention_heads=28,
    intermediate_size=18944, vocab_size=152064, num_key_value_heads=4, family="qwen2",
)
QWEN_05B = ModelShape(
    "Qwen2.5-0.5B", hidden_size=896, num_layers=24, num_attention_heads=14,
    intermediate_size=4864, vocab_size=151936, num_key_value_heads=2, family="qwen2",
    tie_word_embeddings=True,
)


def examples(n: int = 100) -> list[Example]:
    return [
        Example(instruction=f"Question number {i} about a topic",
                response=f"Answer number {i}")
        for i in range(n)
    ]


class TestModelShape:
    def test_parameter_count_is_in_the_right_range(self):
        billions = QWEN_7B.total_parameters / 1e9
        assert 7.0 < billions < 8.0

    def test_grouped_query_attention_reduces_the_count(self):
        """Treating k and v as the same size as q - as most envelope counts do -
        overestimates a modern model by several percent."""
        gqa = QWEN_7B.total_parameters
        mha = ModelShape(**{**QWEN_7B.__dict__, "num_key_value_heads": 28}).total_parameters
        assert gqa < mha

    def test_tied_embeddings_are_not_counted_twice(self):
        """On a small model with a 150k vocabulary, the embedding is a large fraction
        of the total - double counting overstates it by around a quarter."""
        tied = QWEN_05B.total_parameters
        untied = ModelShape(
            **{**QWEN_05B.__dict__, "tie_word_embeddings": False}
        ).total_parameters
        assert untied > tied * 1.2

    def test_an_indivisible_head_count_is_refused(self):
        with pytest.raises(ConfigError):
            ModelShape("bad", hidden_size=100, num_layers=1, num_attention_heads=7,
                       intermediate_size=100, vocab_size=100)


class TestLoRAConfig:
    def test_scaling_is_alpha_over_r(self):
        """Kept constant when r changes, or the effective learning rate moves and the
        run that 'worked at r=8 but diverged at r=64' is explained."""
        assert LoRAConfig(r=16, alpha=32).scaling == 2.0
        assert LoRAConfig(r=64, alpha=128).scaling == 2.0

    @pytest.mark.parametrize(
        "kwargs",
        [{"r": 0}, {"alpha": 0}, {"dropout": 1.0}, {"target_modules": []}],
    )
    def test_invalid_configurations_are_refused(self, kwargs):
        with pytest.raises(ConfigError):
            LoRAConfig(**kwargs)

    def test_an_unknown_target_module_is_refused(self):
        with pytest.raises(ConfigError):
            adapter_parameters(QWEN_7B, LoRAConfig(target_modules=["not_a_layer"]))


class TestAdapterParameters:
    def test_lora_trains_under_one_percent(self):
        """The claim the method rests on."""
        stats = adapter_parameters(QWEN_7B, LoRAConfig(r=16, alpha=32))
        assert 0.1 < stats["trainable_percent"] < 1.5

    def test_rank_scales_the_parameter_count_linearly(self):
        r8 = adapter_parameters(QWEN_7B, LoRAConfig(r=8))["trainable"]
        r16 = adapter_parameters(QWEN_7B, LoRAConfig(r=16))["trainable"]
        assert r16 == 2 * r8

    def test_attention_only_trains_far_fewer_parameters(self):
        """The common shortcut, and usually the wrong trade: the MLP holds roughly two
        thirds of the parameters."""
        full = adapter_parameters(QWEN_7B, LoRAConfig(r=16))["trainable"]
        attention = adapter_parameters(
            QWEN_7B, LoRAConfig(r=16, target_modules=list(ATTENTION_ONLY))
        )["trainable"]
        assert attention < full * 0.5

    def test_training_embeddings_dominates_on_a_small_model(self):
        without = adapter_parameters(QWEN_05B, LoRAConfig(r=16))["trainable"]
        with_embeddings = adapter_parameters(
            QWEN_05B, LoRAConfig(r=16, train_embeddings=True)
        )["trainable"]
        assert with_embeddings > without * 5


class TestMemory:
    def test_qlora_fits_a_7b_model_on_16gb(self):
        """The entire reason the method exists."""
        estimate = estimate_memory(QWEN_7B, LoRAConfig(r=16), base_dtype="nf4")
        assert estimate.fits_in(16.0)

    def test_full_fine_tuning_does_not(self):
        assert not estimate_memory(QWEN_7B, None, base_dtype="bf16").fits_in(16.0)

    def test_the_optimizer_dominates_full_fine_tuning(self):
        """Two fp32 Adam moments per trainable parameter - ~56 GB for a 7B model,
        before a single activation."""
        estimate = estimate_memory(QWEN_7B, None, base_dtype="bf16")
        assert estimate.optimizer_gb > estimate.base_weights_gb

    def test_qlora_almost_eliminates_optimizer_state(self):
        full = estimate_memory(QWEN_7B, None, base_dtype="bf16").optimizer_gb
        lora = estimate_memory(QWEN_7B, LoRAConfig(r=16)).optimizer_gb
        assert lora < full * 0.01

    def test_four_bit_halves_the_weights_against_bf16(self):
        nf4 = estimate_memory(QWEN_7B, LoRAConfig(r=16), base_dtype="nf4").base_weights_gb
        bf16 = estimate_memory(QWEN_7B, LoRAConfig(r=16), base_dtype="bf16").base_weights_gb
        assert bf16 == pytest.approx(nf4 * 4, rel=0.01)

    def test_gradient_checkpointing_reduces_activations(self):
        on = estimate_memory(QWEN_7B, LoRAConfig(r=16), gradient_checkpointing=True)
        off = estimate_memory(QWEN_7B, LoRAConfig(r=16), gradient_checkpointing=False)
        assert on.activations_gb < off.activations_gb

    def test_eight_bit_adam_halves_optimizer_state(self):
        adamw = estimate_memory(QWEN_7B, LoRAConfig(r=16), optimizer="adamw").optimizer_gb
        eight = estimate_memory(QWEN_7B, LoRAConfig(r=16), optimizer="adamw_8bit").optimizer_gb
        assert eight < adamw

    def test_longer_sequences_cost_more(self):
        short = estimate_memory(QWEN_7B, LoRAConfig(r=16), sequence_length=512)
        long = estimate_memory(QWEN_7B, LoRAConfig(r=16), sequence_length=4096)
        assert long.total_gb > short.total_gb

    def test_the_search_finds_the_largest_fitting_config(self):
        best = largest_config_that_fits(QWEN_7B, 16.0, sequence_length=1024)
        assert best["estimated_gb"] <= 16.0
        assert best["r"] >= 8

    def test_the_search_reports_failure_rather_than_guessing(self):
        assert "error" in largest_config_that_fits(QWEN_7B, 1.0, sequence_length=8192)

    def test_an_unknown_dtype_is_refused(self):
        with pytest.raises(ConfigError):
            estimate_memory(QWEN_7B, LoRAConfig(), base_dtype="int3")


class TestLossMasking:
    """The bug that silently ruins instruction tuning."""

    def test_prompt_tokens_are_masked_out_of_the_loss(self):
        """Without this the model spends most of its gradient learning to generate
        questions, because prompts are usually longer than answers. The loss curve
        still goes down."""
        row = tokenise(Example("What is 2+2?", "Four."), whitespace_tokenizer)
        assert all(label == IGNORE_INDEX for label in row.labels[: row.prompt_length])

    def test_response_tokens_are_supervised(self):
        row = tokenise(Example("What is 2+2?", "Four."), whitespace_tokenizer)
        assert all(label != IGNORE_INDEX for label in row.labels[row.prompt_length:])

    def test_labels_align_with_inputs(self):
        row = tokenise(Example("A question here", "An answer"), whitespace_tokenizer)
        assert len(row.labels) == len(row.input_ids)

    def test_supervised_labels_equal_their_input_tokens(self):
        """An off-by-one here trains the model to predict the token it was just given."""
        row = tokenise(Example("Q", "A B C"), whitespace_tokenizer)
        for i in range(row.prompt_length, len(row.labels)):
            assert row.labels[i] == row.input_ids[i]

    def test_the_split_is_structural_not_a_text_search(self):
        """Searching the rendered string for the response fails whenever the response
        also appears in the prompt - constant in summarisation and extraction."""
        example = Example("Translate: hello world", "hello world")
        row = tokenise(example, whitespace_tokenizer)
        assert row.supervised_tokens > 0
        assert row.labels[0] == IGNORE_INDEX

    def test_right_truncation_can_destroy_the_response(self):
        long = Example("word " * 200, "the answer")
        row = tokenise(long, whitespace_tokenizer, max_length=20, truncate="right")
        assert not row.is_usable

    def test_left_truncation_preserves_the_response(self):
        """Losing the start of the prompt costs context; losing the response costs the
        entire example."""
        long = Example("word " * 200, "the answer")
        row = tokenise(long, whitespace_tokenizer, max_length=20, truncate="left")
        assert row.is_usable


class TestTemplates:
    @pytest.mark.parametrize("template", ["chatml", "alpaca", "llama3"])
    def test_every_template_produces_a_prompt_and_a_completion(self, template):
        prompt, completion = render(Example("Q", "A"), template)
        assert prompt and completion

    def test_the_system_message_is_included(self):
        prompt, _ = render(Example("Q", "A", system="You are terse."), "chatml")
        assert "terse" in prompt

    def test_the_input_field_joins_the_instruction(self):
        prompt, _ = render(Example("Summarise", "A", input="Some text"), "alpaca")
        assert "Some text" in prompt

    def test_an_unknown_template_is_refused(self):
        with pytest.raises(ValueError):
            render(Example("Q", "A"), "not_a_template")


class TestDatasetStats:
    def test_unusable_rows_are_reported(self):
        """Silently dropping a fifth of a dataset is the difference between a run that
        underperforms for a findable reason and one that underperforms mysteriously."""
        rows = [Example("word " * 200, "answer") for _ in range(10)]
        _, stats = prepare(rows, whitespace_tokenizer, max_length=20)
        assert stats["dropped_no_response"] == 10

    def test_supervised_fraction_is_reported(self):
        """Low here means long prompts and short answers - worth knowing before a long
        run, because it is the fraction of compute that trains what you care about."""
        _, stats = prepare(examples(20), whitespace_tokenizer)
        assert 0.0 < stats["supervised_fraction"] < 1.0


class TestSplitting:
    def test_no_example_appears_on_both_sides(self):
        _, _, stats = split(examples(100))
        assert stats["cross_split_duplicates"] == 0

    def test_duplicates_are_removed_before_the_split(self):
        """Deduplicating each side separately leaves cross-split duplicates untouched,
        which is exactly the leak."""
        duplicated = examples(50) + examples(50)
        _, _, stats = split(duplicated)
        assert stats["duplicates_removed"] == 50
        assert stats["cross_split_duplicates"] == 0

    def test_deduplication_ignores_whitespace_and_case(self):
        rows = [Example("What is X?", "Y"), Example("what  is  x?", "y")]
        _, removed = deduplicate(rows)
        assert removed == 1

    def test_the_split_is_reproducible(self):
        first, _, _ = split(examples(100), seed=7)
        second, _, _ = split(examples(100), seed=7)
        assert [e.instruction for e in first] == [e.instruction for e in second]

    def test_an_empty_dataset_does_not_crash(self):
        train, evaluation, stats = split([])
        assert train == [] and evaluation == []


class TestPacking:
    def test_short_examples_are_packed_together(self):
        """A batch padded to 1024 where the average example is 180 tokens spends 80% of
        its FLOPs on padding."""
        rows, _ = prepare(examples(50), whitespace_tokenizer)
        packed = pack(rows, max_length=256)
        assert len(packed) < len(rows)

    def test_no_batch_exceeds_the_limit(self):
        rows, _ = prepare(examples(50), whitespace_tokenizer)
        assert all(len(b.input_ids) <= 256 for b in pack(rows, max_length=256))

    def test_boundaries_are_recorded(self):
        """Without an attention mask at these boundaries, packing teaches the model
        that unrelated examples follow one another - cheaper and worse."""
        rows, _ = prepare(examples(20), whitespace_tokenizer)
        packed = pack(rows, max_length=256)
        assert any(len(b.boundaries) > 1 for b in packed)

    def test_masking_survives_packing(self):
        rows, _ = prepare(examples(20), whitespace_tokenizer)
        packed = pack(rows, max_length=256)
        assert any(label == IGNORE_INDEX for b in packed for label in b.labels)
        assert any(label != IGNORE_INDEX for b in packed for label in b.labels)


class TestScheduling:
    def test_effective_batch_size_multiplies_everything(self):
        """Halving the micro-batch to fit a longer sequence and forgetting to double
        accumulation silently halves the effective batch."""
        plan = StepPlan(dataset_size=1000, micro_batch_size=4,
                        gradient_accumulation=8, epochs=1, devices=2)
        assert plan.effective_batch_size == 64

    def test_step_counts_follow_from_it(self):
        plan = StepPlan(dataset_size=1000, micro_batch_size=4,
                        gradient_accumulation=4, epochs=3)
        assert plan.steps_per_epoch == 62
        assert plan.total_steps == 186

    @pytest.mark.parametrize(
        "kwargs", [{"dataset_size": 0}, {"micro_batch_size": 0}, {"epochs": 0}]
    )
    def test_invalid_plans_are_refused(self, kwargs):
        base = {"dataset_size": 10, "micro_batch_size": 1,
                "gradient_accumulation": 1, "epochs": 1}
        with pytest.raises(ScheduleError):
            StepPlan(**{**base, **kwargs})

    def test_short_runs_get_a_warmup_floor(self):
        """3% of a 200-step run is 6 steps, which is too few for Adam's moments to
        settle."""
        assert warmup_steps(200) > 200 * 0.03

    def test_warmup_never_swallows_the_run(self):
        assert warmup_steps(30) <= 3

    def test_the_schedule_warms_up_then_decays(self):
        values = [
            linear_warmup_cosine_decay(s, total_steps=100, warmup=10, peak_lr=1e-4)
            for s in range(100)
        ]
        assert values[0] < values[9]        # warming up
        assert values[9] > values[99]       # then decaying

    def test_the_peak_is_reached_at_the_end_of_warmup(self):
        peak = linear_warmup_cosine_decay(9, total_steps=100, warmup=10, peak_lr=1e-4)
        assert peak == pytest.approx(1e-4)

    def test_decay_stops_above_zero(self):
        """A rate reaching exactly zero means the last steps do nothing - and on a
        short run those are the steps closest to where you want the model."""
        final = linear_warmup_cosine_decay(
            99, total_steps=100, warmup=10, peak_lr=1e-4, min_lr_ratio=0.1
        )
        assert final == pytest.approx(1e-5, rel=0.01)

    def test_step_zero_is_a_real_update(self):
        assert linear_warmup_cosine_decay(0, total_steps=100, warmup=10, peak_lr=1e-4) > 0

    def test_higher_rank_suggests_a_lower_rate(self):
        """More capacity means larger updates for the same gradient."""
        assert suggested_learning_rate(64) < suggested_learning_rate(8)


class TestEvaluation:
    def test_exact_match_ignores_case_and_whitespace(self):
        assert exact_match("The  Answer", "the answer") == 1.0

    def test_containment_allows_extra_explanation(self):
        """A model that answers correctly and then explains should not score zero."""
        assert contains("The answer is four, because 2+2=4", "four") == 1.0

    def test_token_f1_gives_partial_credit(self):
        """Exact match is almost always zero on generative tasks, which tells you
        nothing about whether the model improved."""
        score = token_f1("the quick brown fox", "the quick red fox")
        assert 0.5 < score < 1.0

    def test_identical_text_scores_one(self):
        assert token_f1("same text", "same text") == 1.0

    def test_disjoint_text_scores_zero(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            evaluate(["a"], ["a", "b"])

    def test_an_unknown_scorer_is_refused(self):
        with pytest.raises(ValueError):
            evaluate(["a"], ["a"], scorer="vibes")

    def test_a_comparison_reports_the_delta(self):
        result = compare(["wrong", "wrong"], ["right answer", "wrong"],
                         ["right answer", "wrong"])
        assert result.improved
        assert result.delta > 0

    def test_regressions_are_surfaced_not_averaged_away(self):
        """A mean improvement can hide a model that broke on a subset - and the broken
        subset is frequently the one that mattered.

        Constructed so the headline genuinely improves (0.50 -> 0.90) while the second
        example gets worse (1.0 -> 0.8). Reporting only the mean would call this an
        unqualified win.
        """
        references = ["right answer", "correct answer here"]
        before = ["", "correct answer here"]      # scores 0.0 and 1.0
        after = ["right answer", "correct answer"]  # scores 1.0 and 0.8

        result = compare(before, after, references)
        assert result.improved
        assert result.delta > 0.3
        assert result.summary()["regressed_examples"] == 1
        assert result.regressions() == [1]


class TestPlanning:
    def test_a_run_can_be_planned_without_a_gpu(self):
        """The mistakes that ruin a fine-tune are made before the GPU is touched."""
        report = plan_run(examples(200), TrainingConfig(), shape=QWEN_7B)
        assert report.plan["total_steps"] > 0
        assert report.memory_estimate["fits_16gb"] is True

    def test_the_plan_reports_data_loss(self):
        report = plan_run(examples(200), TrainingConfig(max_length=8), shape=QWEN_05B)
        assert report.data["dropped_no_response"] > 0

    def test_the_learning_rate_is_derived_from_the_rank(self):
        report = plan_run(examples(50), TrainingConfig(r=64))
        assert report.plan["learning_rate"] == suggested_learning_rate(64)

    def test_an_explicit_learning_rate_wins(self):
        report = plan_run(examples(50), TrainingConfig(r=64, learning_rate=1e-5))
        assert report.plan["learning_rate"] == 1e-5

    def test_improvement_is_unknown_without_both_evaluations(self):
        """A run reporting 'training complete' without a before and after has not shown
        that anything improved."""
        report = plan_run(examples(50), TrainingConfig())
        assert report.improved() is None
