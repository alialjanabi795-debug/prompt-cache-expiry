"""Regenerate the paper's figures from results/ -> output/figures/*.pdf.

fig_timeline: one replayed trajectory: local steps, delegated steps, the 5-minute cache lifetime
              after each cloud call, and the calls that found their cache expired
              (results/timeline/, written by scripts/timeline.py)
fig_cost:     key-step cloud cost per trajectory by component, and the same tokens priced without
              a cache (results/figures/fig2_cost_breakdown.csv, measured device model)
fig_expiry:   extra cost from cache expiry vs local decode speed for the three cache lifetimes
              (results/regime_map.csv, measured device model, average network)
Palette: the repo's validated three-slot categorical order (scripts/figures.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "output" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
PRESETS = ["anthropic_5m", "openai_30m", "anthropic_1h"]
LABEL = {"anthropic_5m": "5 min", "openai_30m": "30 min", "anthropic_1h": "1 h"}
COLOR = dict(zip(PRESETS, SERIES, strict=True))
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "lines.linewidth": 1.4,
    }
)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_timeline() -> None:
    d = RES / "timeline"
    steps = pd.read_csv(d / "steps.csv")
    calls = pd.read_csv(d / "calls.csv")
    meta = json.loads((d / "meta.json").read_text())
    ttl_min = meta["ttl_s"] / 60
    m = 60.0
    total = meta["total_s"] / m
    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    y_cloud, y_dev = 1.0, 0.0
    for _, s in steps.iterrows():
        if s.tier == "cloud":
            continue
        color = SERIES[0] if s.tier == "local" else MUTED
        ax.barh(
            y_dev,
            (s.t_llm_end_s - s.t_start_s) / m,
            left=s.t_start_s / m,
            height=0.5,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
    # periods in which the cached prefix is alive: each call (re)starts a lifetime of one TTL
    alive: list[list[float]] = []
    for t0 in calls.t_start_s / m:
        if alive and t0 <= alive[-1][1]:
            alive[-1][1] = t0 + ttl_min
        else:
            alive.append([t0, t0 + ttl_min])
    for a, b in alive:
        ax.barh(
            y_cloud, min(b, total) - a, left=a, height=0.5, color=SERIES[1], alpha=0.22, linewidth=0
        )
    for _, c in calls.iterrows():
        t0 = c.t_start_s / m
        ax.plot([t0, t0], [y_cloud - 0.25, y_cloud + 0.25], color=SERIES[1], lw=1.6)
        if c.expired:
            ax.plot(t0, y_cloud + 0.42, marker="v", ms=6, color=INK)
            left = t0 < total / 2
            ax.annotate(
                f"{c.gap_s / m:.1f}-min gap: cache expired,\n"
                f"{c.expired_tokens / 1000:.1f}k tokens billed again\n"
                f"(\\${c.usd:.2f}, not \\${c.usd_if_never_expired:.2f})",
                (t0, y_cloud + 0.55),
                xytext=(6 if left else -6, 12),
                textcoords="offset points",
                ha="left" if left else "right",
                va="bottom",
                fontsize=6.5,
                color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6},
            )
    ax.set_yticks([y_dev, y_cloud])
    ax.set_yticklabels(["device\n(local steps)", "cloud\n(delegated steps)"])
    ax.set_ylim(-0.45, 2.35)
    ax.set_xlim(0, total * 1.01)
    ax.set_xlabel("time since the start of the task (minutes)")
    ax.grid(axis="y", visible=False)
    handles = [
        Patch(color=SERIES[0], label="local step on the device"),
        Patch(color=MUTED, label="step on the edge server (fallback)"),
        Line2D([], [], color=SERIES[1], lw=1.6, label="delegated step (cloud call)"),
        Patch(color=SERIES[1], alpha=0.22, label="cached prefix alive"),
        Line2D([], [], color=INK, marker="v", lw=0, ms=6, label="call finds its cache expired"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=5,
        fontsize=6.5,
        handlelength=1.4,
        columnspacing=1.0,
    )
    save(fig, "fig_timeline")


def fig_cost() -> None:
    d = pd.read_csv(RES / "figures" / "fig2_cost_breakdown.csv")
    comps = [
        ("output", "output tokens", MUTED),
        ("input at warm-cache price", "input, priced with a warm cache", SERIES[0]),
        ("cold re-prefill excess", "extra cost from cache expiry", SERIES[1]),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    ylab, y = [], 0
    for p in PRESETS:
        for t in (5, 11, 30):
            r = d[(d.preset == p) & (d.decode_tps == t)].iloc[0]
            left = 0.0
            for col, lab, color in comps:
                ax.barh(
                    y,
                    r[col],
                    left=left,
                    color=color,
                    height=0.7,
                    edgecolor="white",
                    lw=0.6,
                    label=lab if y == 0 else None,
                )
                left += r[col]
            nocache = r["no-cache price (b)"]
            ax.plot(
                nocache,
                y,
                marker="|",
                color=INK,
                ms=7,
                mew=1.2,
                label="same tokens, priced without a cache" if y == 0 else None,
            )
            ax.text(
                nocache + 0.05, y, f"{nocache / left:.1f}×", va="center", fontsize=6.5, color=INK2
            )
            ylab.append(f"{LABEL[p]} cache, {t} tok/s")
            y += 1
        y += 0.4
    ax.set_yticks([i + 0.4 * (i // 3) for i in range(9)])
    ax.set_yticklabels(ylab, fontsize=6.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("cloud cost per task (US\\$), key-step policy")
    ax.set_xlim(0, 3.7)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.42, 1.0),
        ncol=2,
        fontsize=6,
        handlelength=1.2,
        columnspacing=0.8,
    )
    save(fig, "fig_cost")


def fig_expiry() -> None:
    rm = pd.read_csv(RES / "regime_map.csv")
    rm = rm[(rm.network == "average") & (rm.arm == "thermal_off")]
    thr = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())["criteria"]["K1"][
        "threshold"
    ]
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    for p in PRESETS:
        d = rm[rm.preset == p].sort_values("decode_tps")
        ax.plot(
            d.decode_tps,
            100 * d.K1_share,
            color=COLOR[p],
            marker="o",
            ms=2.8,
            label=f"cache lifetime {LABEL[p]}",
        )
    ax.axhline(100 * thr, color=MUTED, lw=0.8, ls=":")
    ax.text(
        12,
        100 * thr - 4.5,
        f"{100 * thr:.0f}% threshold",
        ha="left",
        fontsize=6.5,
        color=INK2,
    )
    ax.set_xscale("log")
    ax.set_xticks([5, 11, 30, 60])
    ax.set_xticklabels(["5", "11", "30", "60"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("local model speed on the device (tokens/s)")
    ax.set_ylabel("extra cost from expiry\n(% of the cloud bill)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=6.5,
        handlelength=1.5,
        columnspacing=1.0,
    )
    save(fig, "fig_expiry")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for old in ("fig_regime.pdf", "fig_provider.pdf", "fig_provider.csv"):
        (FIG / old).unlink(missing_ok=True)
    fig_timeline()
    fig_cost()
    fig_expiry()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
