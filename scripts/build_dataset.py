"""Sample, tokenize and normalize trajectories into data/processed/*.parquet.

Usage: python -I scripts/build_dataset.py [--n 2000] [--seed 20261019]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from cache_delegation.loader import iter_raw, normalize, sample_indices, to_frames  # noqa: E402
from cache_delegation.preregistration import sha256_file  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def build(name: str, path: Path, source: str, n: int, seed: int, tok: Tokenizer) -> dict:
    rows = pq.ParquetFile(path).metadata.num_rows
    idx = sample_indices(rows, n, seed)
    t0 = time.time()
    trajs = [normalize(r, tok) for r in iter_raw(path, source, idx)]
    tdf, sdf = to_frames(trajs)
    OUT.mkdir(parents=True, exist_ok=True)
    tdf.to_parquet(OUT / f"{name}_trajectories.parquet", index=False)
    sdf.to_parquet(OUT / f"{name}_steps.parquet", index=False)
    return {
        "name": name,
        "source_file": str(path.relative_to(ROOT)),
        "rows_in_file": rows,
        "sampled": len(idx),
        "kept": len(trajs),
        "seed": seed,
        "file_sha256": sha256_file(path),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=crit["data"]["sample"]["n"])
    ap.add_argument("--seed", type=int, default=crit["data"]["sample"]["seed"])
    ap.add_argument("--swesmith-n", type=int, default=1000)
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(RAW / "tokenizer" / "tokenizer.json"))
    manifest = [
        build(
            "openhands",
            RAW / "openhands" / "trajectories.parquet",
            "openhands",
            args.n,
            args.seed,
            tok,
        ),
        build(
            "swesmith",
            RAW / "swesmith" / "tool-00004-of-00008.parquet",
            "swesmith",
            args.swesmith_n,
            args.seed,
            tok,
        ),
    ]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
