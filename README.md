# prompt-cache-expiry

This repository contains the code, data and results for a replay study of edge–cloud delegation
for coding agents. It measures whether a slow local device, by stretching the time between
cloud calls, lets the provider's prompt cache expire and so raises the cloud bill, and how much
cache-aware pricing and keep-warm requests change the cost.

## Requirements

- Python 3.11 or newer; `pip install -e .[dev]` installs numpy, pandas, pyarrow, pyyaml,
  tokenizers, matplotlib, pytest and ruff.
- About 3 GB of disk for the raw data (only needed to rebuild the task sample).
- Run time: the full pipeline (`scripts/run_all.sh`, 4 worker processes) took 57 minutes on a
  cloud machine with 4 vCPUs (Intel Xeon, 2.1 GHz) and 15 GB of RAM: about 4 minutes for the
  sample, 52 minutes for the simulations and about 1 minute for the analysis, numbers and
  figures.

## Reproduce everything

```sh
pip install -e .[dev]
sh scripts/download_raw.sh        # raw inputs -> data/raw/, SHA-256 checked
WORKERS=4 sh scripts/run_all.sh   # sample, simulations, analysis, numbers, tables, figures
pytest -q
```

`scripts/run_all.sh` runs these steps; each can also be run on its own:

| step | command | writes |
|---|---|---|
| rebuild the 2,000-task sample (seed 20261019) | `python -I scripts/build_dataset.py` | `data/processed/` |
| run the simulations | `python -I scripts/run_sweep.py --grid all --workers 4` | `results/sweep/` |
| analyse | `python -I scripts/analyze.py` | `results/*.csv`, `results/verdicts.json` |
| add the live provider check | `python -I scripts/calib/settle_k2.py` | `results/verdicts.json` |
| figure data | `python -I scripts/figures.py` | `results/figures/*.csv` |
| keep-warm cost with measured request output | `python -I scripts/reprice_pings.py` | `results/k2_repriced.*` |
| one replayed task (timeline) | `python -I scripts/timeline.py` | `results/timeline/` |
| every reported number and table | `python -I scripts/paper_numbers.py` | `output/numbers.json`, `output/numbers.tex`, `output/table_*.tex` |
| every figure | `python -I scripts/paper_figures.py` | `output/figures/*.pdf` |

The processed sample and all results are already included, so the last two commands work
without the raw data or the simulations.

### Quick check (a few minutes)

```sh
python -I scripts/run_sweep.py --grid all --limit 50 --out-dir results/sweep_smoke  # 50 tasks
python -I scripts/paper_numbers.py && python -I scripts/paper_figures.py           # from results/
```

The first command runs every simulation grid on the first 50 tasks only; the second regenerates
every reported number, table and figure from the results files in this repository.

## Pre-registration

The decision rules were written in `configs/kill_criteria.yaml` before any simulation on real
tasks, and the file's SHA-256 hash was recorded in `configs/kill_criteria.lock.json`. Every
simulation and analysis script refuses to run if the file does not match. To check it yourself:

```sh
sha256sum configs/kill_criteria.yaml
grep sha256 configs/kill_criteria.lock.json   # both must show c81c4b40…89207e
```

Every later change to inputs, reporting or code is logged with its reason in
`docs/deviations.md`. In this public copy the module that checks the hash is
`src/cache_delegation/preregistration.py` (named `killcriteria.py` in the development history
that the log refers to).

## Live provider check

`scripts/calib/provider_check.py --backend openrouter` re-runs the check of cache lifetimes and
prices on the live APIs. It needs an OpenRouter key in `PROVIDER_CHECK_OPENROUTER_KEY` and costs
a few US dollars. The recorded requests (token counts and costs only) are in `results/calib/`.

## Device calibration

`results/calib/` holds the measurements used to calibrate the device model. The scripts in
`scripts/calib/` (llama.cpp server, agent replay, memory-pressure cases) re-run them; they need a
llama.cpp build (`LLAMA_SERVER`), the Qwen3.5-9B Q4_K_M weights (`MODEL_GGUF`) and Linux memory
cgroups, and are not needed to reproduce the reported results.

## Layout

```
configs/   device, network, provider price and tool-time settings; pre-registered rules + lock
data/processed/   the 2,000-task sample and the 1,000-task SWE-smith set (derived data)
docs/deviations.md   log of every change after the rules were frozen
results/   simulation summaries, calibration data, provider-check records
scripts/   dataset, simulation, analysis, numbers and figures
src/cache_delegation/   simulator, device, cache, pricing and policy models
tests/     unit and pipeline tests
```

## Third-party data and licences

- `data/processed/openhands_*.parquet` is derived from **SWE-rebench OpenHands trajectories**
  by Nebius (https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories),
  licensed **CC BY 4.0**. Changes: a uniform random sample of 2,000 runs (seed 20261019),
  normalised into steps, with token counts added.
- `data/processed/swesmith_*.parquet` is derived from **SWE-smith trajectories**
  (https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories, shard
  `tool-00004-of-00008`), licensed **MIT**. Same kind of changes, 1,000 runs.
- Token counts use the **Qwen3.5-9B** tokenizer (https://huggingface.co/Qwen/Qwen3.5-9B),
  licensed **Apache-2.0**; the tokenizer itself is not redistributed here
  (`scripts/download_raw.sh` fetches it).
- Raw inputs are not included; `scripts/download_raw.sh` downloads them and checks their hashes.

Our code is under the MIT licence (`LICENSE`); our results are under CC BY 4.0
(`LICENSE-results.md`).
