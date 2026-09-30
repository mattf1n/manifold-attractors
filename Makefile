STYLE ?= iclr
.PHONY: install plan paper figures tables check test smoke
install:
	uv sync --locked --extra compute
plan:
	uv run reproduce plan --style $(STYLE)
paper:
	uv run --extra compute reproduce run paper --style $(STYLE) --resume
figures:
	uv run --extra compute reproduce run figures --style $(STYLE) --resume
tables:
	uv run --extra compute reproduce run tables --resume
check:
	uv run reproduce check --style $(STYLE)
test:
	uv run --extra compute --extra render --group dev pytest -q
	uv run --locked --isolated --extra table --group dev pytest tests/test_tables.py -q
smoke:
	uv run reproduce run render-dead_fish_sequence render-two_basin_field --style $(STYLE)
