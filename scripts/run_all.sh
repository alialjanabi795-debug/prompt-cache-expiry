#!/bin/sh
# Full pipeline: sample -> simulations -> analysis -> every number, table and figure in output/.
# WORKERS (default 4) sets the number of simulation processes.
set -eu
cd "$(dirname "$0")/.."
W=${WORKERS:-4}
sha256sum configs/kill_criteria.yaml            # must equal the hash in configs/kill_criteria.lock.json
python3 -I scripts/build_dataset.py             # 2,000-task sample (fixed seed) -> data/processed/
python3 -I scripts/run_sweep.py --grid all --workers "$W"   # simulations -> results/sweep/
python3 -I scripts/analyze.py                   # results/*.csv, results/verdicts.json
python3 -I scripts/calib/settle_k2.py           # adds the live provider check to verdicts.json
python3 -I scripts/figures.py                   # results/figures/*.csv (figure data)
python3 -I scripts/reprice_pings.py             # results/k2_repriced.*
python3 -I scripts/timeline.py                  # results/timeline/ (one replayed task)
python3 -I scripts/paper_numbers.py             # output/numbers.json, numbers.tex, table_*.tex
python3 -I scripts/paper_figures.py             # output/figures/*.pdf
