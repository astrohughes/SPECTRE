#!/usr/bin/env python3
"""Visualization tools for SPECTRE maneuver detection results.
TODO: Make this WAY better

Generates publication-quality plots of orbital element evolution with
detected maneuvers highlighted. Supports both interactive analysis
(Jupyter) and batch report generation (saved PNGs).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch


# Use a clean style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 10,
})

# Maneuver type colors
TYPE_COLORS = {
    "ALTITUDE_RAISE": "#2ecc71",
    "ALTITUDE_LOWER": "#e74c3c",
    "ALTITUDE_MAINTENANCE": "#3498db",
    "PLANE_CHANGE": "#9b59b6",
    "PHASING": "#f39c12",
    "ORBIT_RAISE": "#1abc9c",
    "DEORBIT": "#e74c3c",
    "UNKNOWN": "#95a5a6",
}


def plot_element_history(
    element_df: pd.DataFrame,
    maneuver_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Plot orbital element evolution with detected maneuvers.

    Args:
        element_df: DataFrame from build_element_history()
        maneuver_df: DataFrame from detect_maneuvers_batch() (optional)
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    epochs = element_df["epoch"]

    # Panel 1: Altitude
    ax = axes[0]
    ax.plot(epochs, element_df["altitude_km"], linewidth=0.8, color="#2c3e50")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(title or f"NORAD {element_df['norad_id'].iloc[0]} — Orbital Element History")

    # Panel 2: Inclination
    ax = axes[1]
    ax.plot(epochs, element_df["inclination_deg"], linewidth=0.8, color="#8e44ad")
    ax.set_ylabel("Inclination (°)")

    # Panel 3: RAAN
    ax = axes[2]
    ax.plot(epochs, element_df["raan_deg"], linewidth=0.8, color="#2980b9")
    ax.set_ylabel("RAAN (°)")

    # Panel 4: Eccentricity
    ax = axes[3]
    ax.plot(epochs, element_df["eccentricity"], linewidth=0.8, color="#e67e22")
    ax.set_ylabel("Eccentricity")
    ax.set_xlabel("Epoch")

    # Overlay maneuver markers
    if maneuver_df is not None and not maneuver_df.empty:
        for _, mnvr in maneuver_df.iterrows():
            color = TYPE_COLORS.get(mnvr["type"], "#95a5a6")
            for ax in axes:
                ax.axvline(
                    mnvr["epoch_after"],
                    color=color,
                    alpha=0.6,
                    linewidth=1.5,
                    linestyle="--",
                )

        # Add legend for maneuver types
        from matplotlib.lines import Line2D
        seen_types = maneuver_df["type"].unique()
        legend_elements = [
            Line2D([0], [0], color=TYPE_COLORS.get(t, "#95a5a6"),
                   linestyle="--", linewidth=1.5, label=t.replace("_", " ").title())
            for t in seen_types
        ]
        axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_maneuver_timeline(
    maneuver_df: pd.DataFrame,
    title: str = "Maneuver Timeline",
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot a timeline of detected maneuvers, color-coded by type.

    Y-axis: altitude change (km). Marker size: estimated delta-v.
    """
    if maneuver_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No maneuvers detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#95a5a6")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    for mtype, color in TYPE_COLORS.items():
        mask = maneuver_df["type"] == mtype
        subset = maneuver_df[mask]
        if subset.empty:
            continue

        sizes = np.clip(subset["estimated_dv_ms"] * 5, 10, 200)
        ax.scatter(
            subset["epoch_after"],
            subset["delta_alt_km"],
            c=color,
            s=sizes,
            alpha=0.7,
            label=mtype.replace("_", " ").title(),
            edgecolors="white",
            linewidths=0.5,
        )

    ax.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax.set_ylabel("Altitude Change (km)")
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_constellation_activity(
    maneuver_df: pd.DataFrame,
    bin_days: int = 7,
    title: str = "Constellation Maneuver Activity",
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Histogram of maneuver activity over time (maneuvers per week).
    """
    if maneuver_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No maneuvers detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#95a5a6")
        return fig

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

    # Top: maneuver count per time bin
    ax = axes[0]
    epochs = pd.to_datetime(maneuver_df["epoch_after"])
    start = epochs.min()
    end = epochs.max()

    bins = pd.date_range(start, end + pd.Timedelta(days=bin_days), freq=f"{bin_days}D")

    # Color-stack by type
    type_counts = {}
    for mtype in TYPE_COLORS:
        mask = maneuver_df["type"] == mtype
        if mask.any():
            subset_epochs = epochs[mask]
            counts, _ = np.histogram(subset_epochs.values.astype(np.int64),
                                     bins=bins.values.astype(np.int64))
            type_counts[mtype] = counts

    if type_counts:
        bottom = np.zeros(len(bins) - 1)
        for mtype, counts in type_counts.items():
            ax.bar(
                bins[:-1],
                counts,
                width=bin_days,
                bottom=bottom,
                color=TYPE_COLORS[mtype],
                label=mtype.replace("_", " ").title(),
                alpha=0.8,
            )
            bottom += counts

    ax.set_ylabel(f"Maneuvers per {bin_days} days")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncols=3)

    # Bottom: cumulative delta-v
    ax = axes[1]
    sorted_df = maneuver_df.sort_values("epoch_after")
    cum_dv = sorted_df["estimated_dv_ms"].cumsum()
    ax.fill_between(
        pd.to_datetime(sorted_df["epoch_after"]),
        cum_dv,
        alpha=0.3,
        color="#2c3e50",
    )
    ax.plot(pd.to_datetime(sorted_df["epoch_after"]), cum_dv,
            linewidth=1, color="#2c3e50")
    ax.set_ylabel("Cumulative Δv (m/s)")
    ax.set_xlabel("Date")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_altitude_vs_time_multi(
    tle_dict: dict[int, list],
    maneuver_df: Optional[pd.DataFrame] = None,
    title: str = "Constellation Altitude Profile",
    max_sats: int = 50,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot altitude over time for multiple satellites overlaid.
    Useful for seeing constellation-wide patterns (e.g., orbit raising campaigns).
    """
    fig, ax = plt.subplots(figsize=figsize)

    sat_ids = list(tle_dict.keys())[:max_sats]

    for norad_id in sat_ids:
        tles = tle_dict[norad_id]
        epochs = [t.epoch_dt for t in tles]
        alts = [t.altitude for t in tles]
        ax.plot(epochs, alts, linewidth=0.4, alpha=0.5, color="#2c3e50")

    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Date")
    ax.set_title(f"{title} ({len(sat_ids)} satellites)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_report(
    maneuver_df: pd.DataFrame,
    element_dfs: Optional[dict[int, pd.DataFrame]] = None,
    output_dir: str | Path = "data/reports",
    constellation_name: str = "Unknown",
) -> Path:
    """Generate a complete analysis report with all plots saved as PNGs.

    Returns the output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary stats
    n_maneuvers = len(maneuver_df)
    n_sats = maneuver_df["norad_id"].nunique() if n_maneuvers else 0
    total_dv = maneuver_df["estimated_dv_ms"].sum() if n_maneuvers else 0

    summary = (
        f"# SPECTRE — Maneuver Report\n"
        f"## {constellation_name}\n\n"
        f"- **Maneuvers detected:** {n_maneuvers}\n"
        f"- **Satellites with maneuvers:** {n_sats}\n"
        f"- **Total estimated Δv:** {total_dv:.1f} m/s\n"
        f"- **Report generated:** {datetime.now():%Y-%m-%d %H:%M UTC}\n\n"
    )

    if n_maneuvers:
        type_counts = maneuver_df["type"].value_counts()
        summary += "### Maneuver Type Distribution\n"
        for mtype, count in type_counts.items():
            summary += f"- {mtype.replace('_', ' ').title()}: {count}\n"

    (output_dir / "report.md").write_text(summary)

    # Generate plots
    if n_maneuvers:
        plot_maneuver_timeline(
            maneuver_df,
            title=f"{constellation_name} — Maneuver Timeline",
            save_path=output_dir / "timeline.png",
        )
        plot_constellation_activity(
            maneuver_df,
            title=f"{constellation_name} — Activity",
            save_path=output_dir / "activity.png",
        )

    plt.close("all")
    return output_dir
