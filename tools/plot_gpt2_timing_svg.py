
#!/usr/bin/env python3
"""Render GPT-2 timing logs into a simple SVG line chart.

This script parses `session.run.total: <ms>` lines from one or more miniort logs
and writes a standalone SVG. It has no third-party dependencies.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIME_RE = re.compile(r"session\.run\.total:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")


@dataclass
class Series:
    label: str
    values: list[float]
    color: str


PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
]


def parse_times(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TIME_RE.search(line)
        if match:
            values.append(float(match.group(1)))
    return values


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    head, *tail = points
    chunks = [f"M {head[0]:.2f} {head[1]:.2f}"]
    chunks.extend(f"L {x:.2f} {y:.2f}" for x, y in tail)
    return " ".join(chunks)


def render_svg(series_list: list[Series], title: str, x_label: str, y_label: str) -> str:
    width = 1100
    height = 640
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_values = [v for series in series_list for v in series.values]
    if not all_values:
        raise ValueError("No timing values found")

    y_min = 0.0
    y_max = max(all_values)
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.1

    max_len = max(len(series.values) for series in series_list)
    x_max = max(1, max_len - 1)

    def sx(idx: float) -> float:
        return margin_left + (idx / x_max) * plot_w if x_max else margin_left + plot_w / 2

    def sy(val: float) -> float:
        return margin_top + plot_h - ((val - y_min) / (y_max - y_min)) * plot_h

    def grid_x(step: int) -> Iterable[int]:
        return range(0, x_max + 1, step)

    def grid_y(step: float) -> Iterable[float]:
        cur = 0.0
        while cur <= y_max + 1e-9:
            yield cur
            cur += step

    x_step = 5 if x_max >= 10 else 1
    y_step = max(25.0, math.ceil(y_max / 8.0 / 5.0) * 5.0)

    out: list[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append("<defs>")
    out.append(
        '<style><![CDATA['
        "text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;}"
        ".title{font-size:26px;font-weight:700;fill:#111827;}"
        ".axis{font-size:13px;fill:#374151;}"
        ".tick{font-size:12px;fill:#6b7280;}"
        ".legend{font-size:13px;fill:#111827;}"
        "]]></style>"
    )
    out.append("</defs>")
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    out.append(f'<text class="title" x="{margin_left}" y="36">{escape(title)}</text>')

    # Plot area background
    out.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#e5e7eb"/>'
    )

    # Grid lines and ticks
    for x in grid_x(x_step):
        px = sx(x)
        out.append(f'<line x1="{px:.2f}" y1="{margin_top}" x2="{px:.2f}" y2="{margin_top + plot_h}" stroke="#e5e7eb"/>')
        out.append(f'<text class="tick" x="{px:.2f}" y="{margin_top + plot_h + 22}" text-anchor="middle">{x}</text>')

    for y in grid_y(y_step):
        py = sy(y)
        out.append(f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left + plot_w}" y2="{py:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text class="tick" x="{margin_left - 12}" y="{py + 4:.2f}" text-anchor="end">{y:.0f}</text>')

    # Axes labels
    out.append(f'<text class="axis" x="{margin_left + plot_w / 2}" y="{height - 28}" text-anchor="middle">{escape(x_label)}</text>')
    out.append(
        f'<text class="axis" x="22" y="{margin_top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 22 {margin_top + plot_h / 2})">{escape(y_label)}</text>'
    )

    # Series lines and points
    legend_x = margin_left + plot_w - 220
    legend_y = margin_top + 18
    legend_item_h = 22
    for idx, series in enumerate(series_list):
        color = series.color
        points = [(sx(i), sy(v)) for i, v in enumerate(series.values)]
        if len(points) > 1:
            out.append(
                f'<path d="{line_path(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for x, y in points:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.2"/>')
        ly = legend_y + idx * legend_item_h
        out.append(f'<rect x="{legend_x}" y="{ly - 10}" width="14" height="14" fill="{color}" rx="2"/>')
        out.append(
            f'<text class="legend" x="{legend_x + 20}" y="{ly}">{escape(series.label)}</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


def parse_series_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid series spec: {spec!r}, expected label=path")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not label:
            raise SystemExit(f"invalid series label in spec: {spec!r}")
        parsed.append((label, path))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Render GPT-2 miniort timing logs to SVG.")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Series spec in the form label=path/to/log. Repeatable.",
    )
    parser.add_argument("--title", default="miniONNXRuntime GPT-2 timing")
    parser.add_argument("--x-label", default="decode step")
    parser.add_argument("--y-label", default="session.run.total (ms)")
    parser.add_argument("--output", required=True, help="Output SVG path")
    args = parser.parse_args()

    if not args.series:
        raise SystemExit("at least one --series label=path is required")

    series_specs = parse_series_specs(args.series)
    series_list: list[Series] = []
    for idx, (label, path) in enumerate(series_specs):
        values = parse_times(path)
        if not values:
            raise SystemExit(f"no session.run.total values found in {path}")
        series_list.append(Series(label=label, values=values, color=PALETTE[idx % len(PALETTE)]))

    svg = render_svg(series_list, args.title, args.x_label, args.y_label)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
