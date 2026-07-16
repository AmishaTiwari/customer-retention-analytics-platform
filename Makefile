# These entry points mirror the staged data flow from the ML System Design:
# raw data -> SQL prep -> modeling dataset -> features -> training -> evaluation -> inference.
#
# As of this commit, the modules they call are documentation-only stubs, so
# each target is currently non-functional — it will become real once the
# commit that implements that stage is done.

.PHONY: install ingest prepare train evaluate score test reproduce

install:
	uv sync --extra dev

ingest:
	python -m retention_platform.data.ingest

prepare:
	python -m retention_platform.data.prepare

train:
	python -m retention_platform.models.train

evaluate:
	python -m retention_platform.evaluation.compare

score:
	python -m retention_platform.inference.score

test:
	pytest tests/ -v

reproduce: ingest prepare train evaluate
	@echo "Reproduction pipeline complete."
