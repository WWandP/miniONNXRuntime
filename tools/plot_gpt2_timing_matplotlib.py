#!/usr/bin/env python3
"""Plot GPT-2 miniort timing logs with Matplotlib.

The script reads `session.run.total: <ms>` entries from one or more log files
and renders a line chart for comparing generation traces.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


TIME_RE = re.compile(r"session\.run\.total:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")


def parse_times(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TIME_RE.search(line)
        if match:
            values.append(float(match.group(1)))
    return values


def sample_step_ticks(length: int) -> list[int]:
    if length <= 8:
        return list(range(length))
    if length <= 20:
        return list(range(0, length, 2))
    step = max(1, round(length / 10))
    ticks = list(range(0, length, step))
    if ticks[-1] != length - 1:
        ticks.append(length - 1)
    return ticks


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.color": "#d1d5db",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
        }
    )


def parse_series(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"invalid series spec: {spec!r}, expected label=path")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise SystemExit(f"invalid series label in spec: {spec!r}")
    return label, path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot miniort GPT-2 timing logs with Matplotlib.")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Series spec in the form label=path/to/log. Repeatable.",
    )
    parser.add_argument("--title", default="miniONNXRuntime GPT-2 timing")
    parser.add_argument("--x-label", default="generation step")
    parser.add_argument("--y-label", default="session.run.total (ms)")
    parser.add_argument("--output", required=True, help="Output image path, usually .png")
    args = parser.parse_args()

    if not args.series:
        raise SystemExit("at least one --series label=path is required")

    parsed = [parse_series(spec) for spec in args.series]
    series_data: list[tuple[str, list[float]]] = []
    for label, path in parsed:
        values = parse_times(path)
        if not values:
            raise SystemExit(f"no session.run.total values found in {path}")
        series_data.append((label, values))

    set_paper_style()

    if len(series_data) != 2:
        raise SystemExit("this plotter expects exactly two series for on/off comparison")

    (label_a, values_a), (label_b, values_b) = series_data
    max_len = max(len(values_a), len(values_b))
    common_len = min(len(values_a), len(values_b))

    colors = {
        label_a: "#1f77b4",
        label_b: "#ff7f0e",
    }

    fig = plt.figure(figsize=(12.5, 8.4), dpi=220)
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.1], hspace=0.16, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)

    def draw_series(axis, label: str, values: list[float], color: str) -> None:
        xs = list(range(len(values)))
        # Keep a very light guide line, but let the points dominate visually.
        axis.plot(
            xs,
            values,
            linewidth=1.2,
            color=color,
            alpha=0.22,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
        axis.scatter(
            xs,
            values,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            label=label,
            zorder=3,
        )
        if values:
            axis.scatter([0], [values[0]], s=110, color=color, marker="s", edgecolor="white", linewidth=1.2, zorder=4)
            axis.annotate(
                f"{values[0]:.0f} ms",
                xy=(0, values[0]),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=10,
                color=color,
                weight="semibold",
            )
            axis.annotate(
                f"{values[-1]:.0f} ms",
                xy=(xs[-1], values[-1]),
                xytext=(8, -12),
                textcoords="offset points",
                fontsize=10,
                color=color,
                weight="semibold",
            )

    draw_series(ax, label_a, values_a, colors[label_a])
    draw_series(ax, label_b, values_b, colors[label_b])

    ax.set_ylabel(args.y_label)
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0.12)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.8, columnspacing=1.8, scatterpoints=1)
    ax.set_xlim(-0.5, max_len - 0.5)
    ax.set_ylim(0, max(max(values_a), max(values_b)) * 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.axvspan(-0.5, 0.5, color="#f3f4f6", alpha=0.85, zorder=0)
    ax.text(
        0.01,
        0.95,
        "prefill",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#4b5563",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", lw=0.8),
    )

    # Ratio panel: how much slower/faster label_a is compared to label_b.
    ratio = [va / vb if vb else float("nan") for va, vb in zip(values_a[:common_len], values_b[:common_len])]
    ratio_x = list(range(common_len))
    ax_ratio.plot(ratio_x, ratio, color="#111827", linewidth=1.2, alpha=0.22, zorder=1)
    ax_ratio.scatter(
        ratio_x,
        ratio,
        s=44,
        color="#111827",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
        zorder=3,
    )
    ax_ratio.axhline(1.0, color="#9ca3af", linestyle="--", linewidth=1.1)
    ax_ratio.fill_between(ratio_x, ratio, [1.0] * len(ratio_x), color="#93c5fd", alpha=0.18)
    ax_ratio.set_ylabel("speedup\n(off/on)")
    ax_ratio.set_xlabel(args.x_label)
    ax_ratio.grid(axis="y")
    ax_ratio.grid(axis="x", alpha=0.10)
    ax_ratio.spines["top"].set_visible(False)
    ax_ratio.spines["right"].set_visible(False)
    ax_ratio.spines["left"].set_linewidth(1.0)
    ax_ratio.spines["bottom"].set_linewidth(1.0)

    if ratio:
        ymin = min(ratio + [1.0]) * 0.9
        ymax = max(ratio + [1.0]) * 1.08
        ax_ratio.set_ylim(ymin, ymax)
        ax_ratio.annotate(f"{ratio[-1]:.2f}×", xy=(ratio_x[-1], ratio[-1]), xytext=(8, 8),
                          textcoords="offset points", fontsize=10, color="#111827", weight="semibold")

    ax_ratio.set_xticks(sample_step_ticks(max_len))
    ax_ratio.set_xlim(-0.5, max_len - 0.5)
    fig.suptitle("GPT-2 KV cache on/off comparison", y=0.98, fontsize=18, fontweight="semibold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08, hspace=0.18)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
