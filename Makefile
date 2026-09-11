.DEFAULT_GOAL := help
SHELL := /bin/sh

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the core only - no GPU stack
	uv sync --group dev

install-train:  ## Install torch, transformers, peft, bitsandbytes
	uv sync --group dev --extra train

test:  ## Run the suite - no GPU, no model, no download
	uv run pytest -q

plan:  ## Estimate VRAM and steps for a config, without touching the GPU
	uv run python scripts/plan.py

lint:  ## Lint
	uv run ruff check src tests scripts
	uv run ruff format --check src tests

fmt:  ## Auto-format
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

.PHONY: help install install-train test plan lint fmt
