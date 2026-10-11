from __future__ import annotations

from pathlib import Path
from typing import Any

# Categorical slots 1-4 from the validated reference palette (validate_palette.js: all checks pass;
# slots 3-4 are below 3:1 on the surface, so every bar carries a value label and summary.md has the table).
SETUP_COLORS = {"base": "#2a78d6", "base_rag": "#eb6834", "ft": "#1baf7a", "ft_rag": "#eda100"}
SURFACE = "#fcfcfb"
TEXT = "#1f1f1e"
MUTED = "#5f5e57"
GRID = "#e4e3dc"

PANELS: list[tuple[str, str, Any, bool]] = [
    ("correctness", "Correctness (0–2)", lambda m: m["correctness"], False),
    ("grounding", "Grounding", lambda m: m["grounding"], True),
    ("code", "Code: TSan + Helgrind clean", lambda m: _both_clean(m), True),
    ("bug_f1", "Bug-finding F1", lambda m: m["bug"]["f1"], True),
    ("declines", "Correct out-of-scope declines", lambda m: m["abstention"]["correct_decline"], True),
    ("ttft", "Time to first token p50 (s)", lambda m: (m["speed"]["ttft_p50"] or 0) / 1000 or None, False),
]


def _both_clean(m: dict[str, Any]) -> float | None:
    t, h = m["code"]["tsan_clean"], m["code"]["helgrind_clean"]
    return min(t, h) if t is not None and h is not None else None


def make_plots(data: dict[str, Any], setups: list[str], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
        }
    )
    S = data["setups"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), facecolor=SURFACE)
    for ax, (_, title, get, pct) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        values = [get(S[s]) for s in setups]
        xs = range(len(setups))
        heights = [0 if v is None else (v * 100 if pct else v) for v in values]
        bars = ax.bar(
            xs, heights, width=0.62, color=[SETUP_COLORS[s] for s in setups], edgecolor=SURFACE, linewidth=2, zorder=3
        )
        for bar, v in zip(bars, values, strict=True):
            label = "n/a" if v is None else (f"{v * 100:.0f}%" if pct else f"{v:.2f}")
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT,
            )
        ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
        ax.set_xticks(list(xs), setups)
        ax.set_ylim(0, (100 if pct else max([h for h in heights if h] + [2 if "Correct" in title else 1])) * 1.15)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SETUP_COLORS[s]) for s in setups]
    fig.legend(handles, setups, loc="upper center", ncol=len(setups), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("OSlm: four-setup comparison", x=0.01, ha="left", y=1.04, fontsize=13, color=TEXT)
    fig.tight_layout()
    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    probe_path = out_dir / "probes.png"
    cats = sorted({c for s in setups for c in data["probes"].get(s, {}).get("by_category", {})})
    if cats:
        fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        width = 0.8 / len(setups)
        for i, s in enumerate(setups):
            vals = [data["probes"][s]["by_category"].get(c, 0) * 100 for c in cats]
            ax.bar(
                [x + i * width for x in range(len(cats))],
                vals,
                width=width,
                color=SETUP_COLORS[s],
                edgecolor=SURFACE,
                linewidth=2,
                label=s,
                zorder=3,
            )
        ax.axhline(80, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.annotate("80% target", (len(cats) - 0.5, 81), fontsize=9, color=MUTED, ha="right")
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(cats))], cats, rotation=35, ha="right")
        ax.set_ylim(0, 110)
        ax.set_ylabel("pass rate (%)")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, ncol=len(setups), loc="upper left", bbox_to_anchor=(0, 1.12))
        ax.set_title(
            "Probe suite pass rate by category (exact values in probes_summary.md)", loc="left", pad=28, color=TEXT
        )
        fig.tight_layout()
        fig.savefig(probe_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        return [path, probe_path]
    return [path]
