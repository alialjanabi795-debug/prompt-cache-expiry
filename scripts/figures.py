"""Paper figures from results/*.csv and results/sweep/*.parquet -> results/figures/*.{png,svg}.

Palette: reference categorical slots 1-3 (blue, orange, aqua), validated all-pairs for three
series; thresholds and grids in muted ink. Solid = measured arm (thermal off), dashed = assumed
thermal on (sensitivity, D5). Every figure has a CSV next to it (table view).

Usage: python -I scripts/figures.py [results_dir] [sweep_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
SWEEP = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "sweep"
FIG = RES / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
PRESETS = ["anthropic_5m", "openai_30m", "anthropic_1h"]
PRESET_LABEL = {
    "anthropic_5m": "TTL 5 min (Anthropic)",
    "openai_30m": "TTL 30 min (OpenAI)",
    "anthropic_1h": "TTL 1 h (Anthropic)",
}
NETS = ["good", "average", "weak"]
NET_LABEL = {
    "good": "good network (100 Mbps, 30 ms)",
    "average": "average (20 Mbps, 80 ms)",
    "weak": "weak (2 Mbps, 250 ms)",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
    }
)


def save(fig, name: str, table: pd.DataFrame | None = None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    if table is not None:
        table.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)


def crossing(x: np.ndarray, y: np.ndarray, thr: float) -> float | None:
    """First x (log-interpolated) where y crosses thr, scanning in increasing x."""
    for i in range(len(x) - 1):
        a, b = y[i] - thr, y[i + 1] - thr
        if a == 0:
            return float(x[i])
        if a * b < 0:
            lx0, lx1 = np.log(x[i]), np.log(x[i + 1])
            return float(np.exp(lx0 + (lx1 - lx0) * a / (a - b)))
    return None


def regime_map() -> None:
    f = RES / "regime_map.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    metrics = [
        ("cold_share", "cold delegations (share)", 0.20, "K5 threshold 0.20"),
        ("K1_share", "cold re-prefill / delegation $  (K1)", 0.15, "K1 threshold 0.15"),
        ("K2_saving", "net saving of optimal pings  (K2)", 0.15, "K2 threshold 0.15"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.4), sharex=True, sharey="row")
    cross_rows = []
    for r, (col, ylab, thr, thr_label) in enumerate(metrics):
        for c, net in enumerate(NETS):
            ax = axes[r, c]
            ax.axhline(thr, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
            if r == 2:
                ax.axhline(0.25, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
            for k, preset in enumerate(PRESETS):
                for arm, ls, alpha in (("thermal_off", "-", 1.0), ("thermal_on", "--", 0.55)):
                    s = d[(d.network == net) & (d.preset == preset) & (d.arm == arm)]
                    s = s.sort_values("decode_tps")
                    if s.empty:
                        continue
                    x, y = s.decode_tps.to_numpy(), s[col].to_numpy()
                    ax.plot(
                        x,
                        y,
                        ls=ls,
                        color=SERIES[k],
                        alpha=alpha,
                        marker="o",
                        markersize=3.5 if arm == "thermal_off" else 0,
                        zorder=3,
                        label=f"{PRESET_LABEL[preset]}" if arm == "thermal_off" else None,
                    )
                    xc = crossing(x, y, thr)
                    cross_rows.append(
                        {
                            "metric": col,
                            "network": net,
                            "preset": preset,
                            "arm": arm,
                            "threshold": thr,
                            "crossing_decode_tps": xc,
                        }
                    )
                    if xc is not None and arm == "thermal_off":
                        ax.plot(
                            [xc],
                            [thr],
                            marker="D",
                            markersize=6,
                            color=SERIES[k],
                            markeredgecolor=SURFACE,
                            markeredgewidth=1.5,
                            zorder=4,
                        )
            ax.set_xscale("log")
            ax.set_xticks([5, 8, 11, 15, 20, 30, 45, 60])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.minorticks_off()
            if r == 0:
                ax.set_title(NET_LABEL[net])
            if c == 0:
                ax.set_ylabel(ylab)
            if c == 2:
                ax.text(
                    1.02,
                    thr,
                    thr_label.replace(" threshold", "\nthreshold"),
                    transform=ax.get_yaxis_transform(),
                    color=INK2,
                    fontsize=7.5,
                    va="center",
                    ha="left",
                    clip_on=False,
                )
                if r == 2:
                    ax.text(
                        1.02,
                        0.25,
                        "STRONG\n0.25",
                        transform=ax.get_yaxis_transform(),
                        color=INK2,
                        fontsize=7.5,
                        va="center",
                        ha="left",
                        clip_on=False,
                    )
            if r == 2:
                ax.set_xlabel("local decode speed (tok/s, log)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    h += [
        plt.Line2D([], [], color=INK2, ls="-"),
        plt.Line2D([], [], color=INK2, ls="--", alpha=0.55),
        plt.Line2D([], [], color=INK2, marker="D", ls="none", markersize=6),
    ]
    lab += [
        "measured arm (thermal off)",
        "assumed thermal on (sensitivity)",
        "threshold crossing (measured arm)",
    ]
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    n = int(d.n_traj.max()) if "n_traj" in d else 0
    fig.suptitle(
        "Regime map: when does the cloud cache go cold, and when does it matter? "
        f"(weak_edge, KeyStep, {n:,} trajectories)",
        fontsize=10.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    save(fig, "fig1_regime_map", pd.DataFrame(cross_rows))
    d.to_csv(FIG / "fig1_regime_map_data.csv", index=False)


def cost_breakdown() -> None:
    f = SWEEP / "k_measured.parquet"
    if not f.exists():
        return
    df = pd.read_parquet(f)
    ks = df[df.policy == "keystep"]
    rows = []
    for (p, tps), g in ks.groupby(["preset", "decode_tps"]):
        out = g.usd_out.sum()
        excess = (g.usd_c_deleg - g.usd_c_inf_ttl_deleg).sum()
        warm_in = g.usd_c_deleg.sum() - out - excess
        rows.append(
            {
                "preset": p,
                "decode_tps": tps,
                "output": out / len(g),
                "input at warm-cache price": warm_in / len(g),
                "cold re-prefill excess": excess / len(g),
                "no-cache price (b)": g.usd_b.mean(),
            }
        )
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    parts = ["output", "input at warm-cache price", "cold re-prefill excess"]
    for ax, tps in zip(axes, sorted(t.decode_tps.unique()), strict=False):
        s = t[t.decode_tps == tps].set_index("preset").loc[PRESETS]
        bottom = np.zeros(len(s))
        for k, part in enumerate(parts):
            ax.bar(
                range(len(s)),
                s[part],
                bottom=bottom,
                width=0.62,
                color=SERIES[k],
                edgecolor=SURFACE,
                linewidth=2,
                label=part,
                zorder=3,
            )
            bottom += s[part].to_numpy()
        ax.scatter(
            range(len(s)),
            s["no-cache price (b)"],
            marker="_",
            s=420,
            color=INK,
            zorder=4,
            label="same tokens without caching (b)",
        )
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(["5 min", "30 min", "1 h"])
        ax.set_title(f"local decode {tps:g} tok/s")
        ax.set_xlabel("cache TTL")
    axes[0].set_ylabel("cloud $ per trajectory (KeyStep)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.suptitle(
        "Where the delegation bill goes: warm input vs cold re-prefill "
        "(weak_edge, measured arm, average network)",
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    save(fig, "fig2_cost_breakdown", t)


def coupling() -> None:
    f = RES / "regime_map.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.network == "average"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    rows = []
    for ax, preset in zip(axes, PRESETS, strict=True):
        off = d[(d.preset == preset) & (d.arm == "thermal_off")].sort_values("decode_tps")
        on = d[(d.preset == preset) & (d.arm == "thermal_on")].sort_values("decode_tps")
        arms = [
            ("constant speed", off.decode_tps, off.cold_share_const, SERIES[0], "-"),
            ("measured slowdown (thermal off)", off.decode_tps, off.cold_share, SERIES[1], "-"),
            ("assumed thermal on", on.decode_tps, on.cold_share, SERIES[2], "--"),
        ]
        for name, x, y, color, ls in arms:
            ax.plot(x, y, color=color, ls=ls, marker="o", label=name, zorder=3)
            rows += [
                {"preset": preset, "arm": name, "decode_tps": a, "cold_share": b}
                for a, b in zip(x, y, strict=True)
            ]
        ax.set_xscale("log")
        ax.set_xticks([5, 8, 11, 15, 20, 30, 45, 60])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_title(PRESET_LABEL[preset])
        ax.set_xlabel("local decode speed (tok/s, log)")
    axes[0].set_ylabel("cold delegations (share)")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle(
        "Coupling: cold-cache share under three device arms (weak_edge, KeyStep, average network)",
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    save(fig, "fig3_coupling_three_arms", pd.DataFrame(rows))


def rankings() -> None:
    f = SWEEP / "k_measured.parquet"
    if not f.exists():
        return
    df = pd.read_parquet(f)
    pols = ["random_0.2", "random_0.4", "keystep", "sticky2_keystep", "all_cloud"]
    sub = df[(df.decode_tps == 11) & df.policy.isin(pols)]
    t = (
        sub.groupby(["preset", "policy"])
        .agg(calls=("calls_a", "mean"), usd_b=("usd_b", "mean"), usd_c=("usd_c", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, preset in zip(axes, PRESETS, strict=True):
        s = t[t.preset == preset].set_index("policy").loc[pols]
        y = np.arange(len(pols))
        ax.scatter(s.usd_b, y, color=SERIES[0], s=40, zorder=3, label="no-cache price (b)")
        ax.scatter(s.usd_c, y, color=SERIES[1], s=40, zorder=3, label="cache-aware price (c)")
        for i, pol in enumerate(pols):
            ax.plot([s.usd_c[pol], s.usd_b[pol]], [i, i], color=GRID, lw=2, zorder=2)
            ax.text(
                s.usd_b[pol],
                i + 0.28,
                f"{s.calls[pol]:.0f} calls",
                fontsize=7,
                color=INK2,
                ha="center",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(pols if preset == PRESETS[0] else [])
        ax.set_title(PRESET_LABEL[preset])
        ax.set_xlabel("cloud $ per trajectory")
    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(
        "Policies by call count vs by price (weak_edge, measured arm, 11 tok/s)",
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    save(fig, "fig4_policy_rankings", t)


if __name__ == "__main__":
    for fn in (regime_map, cost_breakdown, coupling, rankings):
        fn()
        print("done", fn.__name__, file=sys.stderr)
