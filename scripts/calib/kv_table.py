"""KV-cache size of Qwen3.5-9B from its config, cross-checked against llama.cpp's allocation log.

Only the full-attention layers keep a KV cache (8 of 32: full_attention_interval = 4); each has
4 KV heads x head_dim 256 for K and V. The 24 Gated-DeltaNet layers keep a fixed recurrent state.
llama.cpp q8_0 stores 32 values in 34 bytes; q4_0 in 18 bytes.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cfg = json.loads((ROOT / "data/raw/tokenizer/config.json").read_text())["text_config"]
n_full = sum(t == "full_attention" for t in cfg["layer_types"])
elems = 2 * n_full * cfg["num_key_value_heads"] * cfg["head_dim"]  # K and V values per token
bpt = {"f16": elems * 2, "q8_0": elems * 34 // 32, "q4_0": elems * 18 // 32}
# recurrent state per sequence: S = n_layers_linear * v_heads * k_dim * v_dim * 4 B (f32)
n_lin = len(cfg["layer_types"]) - n_full
s_state = (
    n_lin
    * cfg["linear_num_value_heads"]
    * cfg["linear_key_head_dim"]
    * cfg["linear_value_head_dim"]
    * 4
)
out = {
    "layers_total": len(cfg["layer_types"]),
    "layers_full_attention": n_full,
    "kv_heads": cfg["num_key_value_heads"],
    "head_dim": cfg["head_dim"],
    "bytes_per_token": bpt,
    "recurrent_state_mib": s_state / 2**20,
    "llama_cpp_log_check": "16384 cells, 8 layers, f16 -> 512.00 MiB; RS buffer 50.25 MiB",
    "sizes_gib": {
        str(ctx): {k: v * ctx / 2**30 for k, v in bpt.items()} for ctx in (16_384, 53_351, 131_072)
    },
}
(ROOT / "results/calib/kv_table.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
