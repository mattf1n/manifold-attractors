STYLE ?= iclr
.PHONY: install plan figures check test smoke
install:
	uv sync --locked --extra compute
plan:
	uv run reproduce plan --style $(STYLE)
figures:
	uv run --extra compute reproduce run --style $(STYLE) --resume
check:
	uv run reproduce check --style $(STYLE)
test:
	uv run --extra compute --extra render --group dev pytest -q
smoke:
	uv run reproduce run render-dead_fish_sequence render-two_basin_field --style $(STYLE)
