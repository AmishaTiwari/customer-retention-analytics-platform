# Entry points for the staged pipeline:
# ingest -> prepare -> features -> train -> evaluate -> score.
#
# Every stage through scoring is implemented. `evaluate` is a placeholder that
# only prints a notice. `train` really trains, and `make reproduce` stops at
# train if the selected XGBoost hyperparameters differ from the approved
# reference values; that stop is intended.
#
# `make score` uses the newest valid model run; pass RUN_ID=<run_id> to pick one.

.PHONY: install ingest prepare features train evaluate score test reproduce

install:
	uv sync --extra dev

ingest:
	uv run python -m retention_platform.data.ingest

prepare:
	uv run python -m retention_platform.data.prepare

features:
	uv run python -m retention_platform.features.build

train:
	uv run python -m retention_platform.models.train

evaluate:
	@echo "The evaluation stage is not implemented yet and is planned for Commit 12."

score:
	uv run python -m retention_platform.inference.score $(if $(RUN_ID),--run-id $(RUN_ID))

test:
	uv run pytest tests/ -v

reproduce: ingest prepare features train evaluate score
	@echo "Reproduction pipeline complete."
