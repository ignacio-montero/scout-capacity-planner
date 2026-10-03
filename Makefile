# Entry points for the project. Every Python command goes through `uv run`
# (never the system python), so it uses the locked environment in .venv/.

# Dev stage targets write to this fixed run folder (DATA_CONTRACTS.md section 0).
DEV_RUN := data/runs/dev
PARAMS ?= config/default.yaml
SWEEP ?= quick
SWEEP_ARGS ?=

.PHONY: setup test test-all lint format \
        data forecast plan simulate run sweep app charts all

# Placeholder recipe: print a message and fail. $(1) = milestone that implements it.
todo = @echo "make $@: not implemented yet ($(1))" >&2; exit 1

## --- environment and quality ----------------------------------------------------

setup:  ## install the locked environment and enable the repo's git hooks
	uv sync --locked
	git config core.hooksPath .githooks

test:  ## fast test suite (skips tests marked slow; includes the guardrail check)
	uv run pytest -m "not slow"

test-all:  ## full test suite including slow end-to-end tests (~7 min)
	uv run pytest

lint:  ## static checks; fails if code is not formatted
	uv run ruff check .
	uv run ruff format --check .

format:  ## auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

## --- pipeline stages (dev run folder: $(DEV_RUN)) -----------------------------------

data:  ## [1] synthetic world -> $(DEV_RUN)/raw/*.parquet
	uv run python -m scout_planner.cli data --params $(PARAMS) --out $(DEV_RUN)

forecast:  ## [2] demand forecast + backtest -> $(DEV_RUN)/{forecast,backtest}.parquet
	uv run python -m scout_planner.cli forecast --run $(DEV_RUN)

plan:  ## [3] capacity plan + hiring table -> $(DEV_RUN)/{capacity_plan,hiring_plan}.parquet
	uv run python -m scout_planner.cli plan --run $(DEV_RUN)

simulate:  ## [5] simulate the dev run -> $(DEV_RUN)/results/*.parquet + summary.json
	uv run python -m scout_planner.cli simulate --run $(DEV_RUN)

## --- runs, sweeps, app, charts -------------------------------------------------

run:  ## create a run from PARAMS in data/runs/ and execute it now (NAME=... optional)
	uv run python -m scout_planner.cli run --params $(PARAMS) $(if $(NAME),--name "$(NAME)",)

sweep:  ## expand config/sweeps/$(SWEEP).yaml into runs and execute them (SWEEP=quick)
	uv run python -m scout_planner.cli sweep --sweep config/sweeps/$(SWEEP).yaml $(SWEEP_ARGS)

app:  ## start the simulator app and its background worker (Ctrl-C stops both)
	uv run python -m scout_planner.serve

charts:
	$(call todo,M6)

all: test lint  ## tests + lint, the dev pipeline end to end, then the quick sweep
	$(MAKE) data forecast plan simulate PARAMS=$(PARAMS)
	$(MAKE) sweep SWEEP=quick
