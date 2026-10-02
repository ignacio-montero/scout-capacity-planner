# Entry points for the project. Every Python command goes through `uv run`
# (never the system python), so it uses the locked environment in .venv/.

# Dev stage targets write to this fixed run folder (DATA_CONTRACTS.md section 0).
DEV_RUN := data/runs/dev
PARAMS ?= config/default.yaml
SWEEP ?= quick

.PHONY: setup test lint format \
        data forecast plan simulate run sweep app charts all

# Placeholder recipe: print a message and fail. $(1) = milestone that implements it.
todo = @echo "make $@: not implemented yet ($(1))" >&2; exit 1

## --- environment and quality ----------------------------------------------------

setup:  ## install the locked environment and enable the repo's git hooks
	uv sync --locked
	git config core.hooksPath .githooks

test:  ## run the test suite (includes the guardrail check)
	uv run pytest

lint:  ## static checks; fails if code is not formatted
	uv run ruff check .
	uv run ruff format --check .

format:  ## auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

## --- pipeline stages (dev run folder: $(DEV_RUN)) -----------------------------------

data:
	$(call todo,M1)

forecast:
	$(call todo,M2)

plan:
	$(call todo,M2)

simulate:
	$(call todo,M4)

## --- runs, sweeps, app, charts -------------------------------------------------

run:
	$(call todo,M4)

sweep:
	$(call todo,M4)

app:
	$(call todo,M5)

charts:
	$(call todo,M6)

all:
	$(call todo,M6)
