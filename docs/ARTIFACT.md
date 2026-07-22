# Artifact Guide

Operational notes for reproducing `AlgoSkill: Learning to Design Algorithms by Scheduling Human-Like Skills` from the public `algorithm_skill` repository.

## Review Path

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.
- `assets/`: README and paper-facing visual assets.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `backend_config.example.json`: Template for backend/model routing.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
```

If no smoke command is tracked, use the README Quick Start with the smallest seed, sample, or task count.

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `bash scripts/reproduce_hardbench.sh`
- `bash scripts/reproduce_postcutoff.sh`
- `bash scripts/reproduce_v4.sh`

## Figure Assets

- `assets/algoskill_intuition.png`
- `assets/algoskill_pipeline.png`

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
