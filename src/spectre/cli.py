#!/usr/bin/env python3
"""SPECTRE command-line interface.

Usage::

    spectre scan --norad-id 25544 --days 90
    spectre scan --file data/starlink_tles.txt
    spectre constellation --name starlink --days 30
    spectre report --file data/detections.csv
"""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .tle_parser import TLE
from .detector import (
    ManeuverDetector,
    DetectionThresholds,
    ManeuverEvent,
    detect_maneuvers_batch,
    build_element_history,
)
from .spacetrack import SpaceTrackClient, load_tle_file

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool):
    """SPECTRE — Spacecraft Propulsive Event Classification & Tracking from Repeated Elements."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s — %(message)s")


@main.command()
@click.option("--norad-id", "-n", type=int, help="NORAD catalog ID")
@click.option("--file", "-f", "filepath", type=click.Path(exists=True), help="TLE file path")
@click.option("--days", "-d", default=90, help="Days of history to analyze")
@click.option("--threshold", "-t", default="default",
              type=click.Choice(["default", "starlink", "oneweb", "iridium"]),
              help="Detection threshold preset")
@click.option("--output", "-o", type=click.Path(), help="Save results to CSV")
def scan(
    norad_id: int | None,
    filepath: str | None,
    days: int,
    threshold: str,
    output: str | None,
):
    """Scan a single spacecraft for maneuvers."""
    thresholds = _get_thresholds(threshold)

    if filepath:
        tles = load_tle_file(filepath)
        console.print(f"Loaded {len(tles)} TLEs from {filepath}")
    elif norad_id:
        console.print(f"Fetching TLE history for NORAD {norad_id} ({days} days)...")
        client = SpaceTrackClient()
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        tles = client.get_tle_history(norad_id, start, end)
        console.print(f"Fetched {len(tles)} TLEs")
    else:
        console.print("[red]Error: provide --norad-id or --file[/red]")
        sys.exit(1)

    if not tles:
        console.print("[yellow]No TLEs found.[/yellow]")
        return

    # Run detection
    detector = ManeuverDetector(thresholds)
    events = detector.detect(tles)

    # Display results
    _display_results(tles, events)

    # Save if requested
    if output:
        import pandas as pd
        df = pd.DataFrame([e.to_dict() for e in events])
        df.to_csv(output, index=False)
        console.print(f"\nResults saved to {output}")


@main.command()
@click.option("--name", "-n", required=True, help="Constellation name (e.g., 'starlink')")
@click.option("--norad-ids", "-i", type=str, help="Comma-separated NORAD IDs")
@click.option("--id-file", type=click.Path(exists=True), help="File with NORAD IDs (one per line)")
@click.option("--days", "-d", default=30, help="Days of history")
@click.option("--threshold", "-t", default="default",
              type=click.Choice(["default", "starlink", "oneweb", "iridium"]))
@click.option("--output", "-o", type=click.Path(), help="Save results to CSV")
@click.option("--report-dir", type=click.Path(), help="Generate report with plots")
def constellation(
    name: str,
    norad_ids: str | None,
    id_file: str | None,
    days: int,
    threshold: str,
    output: str | None,
    report_dir: str | None,
):
    """Scan an entire constellation for maneuvers."""
    thresholds = _get_thresholds(threshold)

    # Get NORAD IDs
    ids = []
    if norad_ids:
        ids = [int(x.strip()) for x in norad_ids.split(",")]
    elif id_file:
        ids = [int(line.strip()) for line in Path(id_file).read_text().splitlines()
               if line.strip().isdigit()]
    else:
        console.print("[red]Provide --norad-ids or --id-file[/red]")
        sys.exit(1)

    console.print(f"Scanning {len(ids)} satellites in {name} ({days} days)...")

    client = SpaceTrackClient()
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    tle_dict = client.get_constellation_tles(ids, start, end)

    console.print(f"Fetched TLEs for {len(tle_dict)} satellites")

    # Detect maneuvers
    df = detect_maneuvers_batch(tle_dict, thresholds)

    # Summary
    n_events = len(df)
    n_sats = df["norad_id"].nunique() if n_events else 0
    console.print(
        Panel(
            f"[bold]{name.upper()}[/bold]\n"
            f"Maneuvers detected: [bold green]{n_events}[/bold green]\n"
            f"Spacecraft with maneuvers: {n_sats}/{len(ids)}\n"
            f"Total estimated Δv: {df['estimated_dv_ms'].sum():.1f} m/s"
            if n_events else f"No maneuvers detected in {days} day window",
            title="Constellation Scan Results",
            box=box.ROUNDED,
        )
    )

    if n_events:
        _display_maneuver_table(df)

    if output:
        df.to_csv(output, index=False)
        console.print(f"\nResults saved to {output}")

    if report_dir:
        from .viz import generate_report
        path = generate_report(df, constellation_name=name.upper(), output_dir=report_dir)
        console.print(f"Report generated in {path}")


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
def analyze(filepath: str):
    """Analyze a previously saved detections CSV."""
    import pandas as pd

    df = pd.read_csv(filepath, parse_dates=["epoch_before", "epoch_after"])
    console.print(f"Loaded {len(df)} detections from {filepath}")
    _display_maneuver_table(df)


def _get_thresholds(name: str) -> DetectionThresholds:
    presets = {
        "default": DetectionThresholds(),
        "starlink": DetectionThresholds.for_starlink(),
        "oneweb": DetectionThresholds.for_oneweb(),
        "iridium": DetectionThresholds.for_iridium(),
    }
    return presets[name]


def _display_results(tles: list[TLE], events: list[ManeuverEvent]):
    """Display scan results with rich formatting."""
    tle = tles[0]
    console.print(
        Panel(
            f"[bold]{tle.name or 'UNKNOWN'}[/bold] (NORAD {tle.norad_id})\n"
            f"TLEs analyzed: {len(tles)}\n"
            f"Time span: {tles[0].epoch_dt:%Y-%m-%d} → {tles[-1].epoch_dt:%Y-%m-%d}\n"
            f"Altitude: {tle.altitude:.1f} km\n"
            f"Inclination: {tle.inclination:.2f}°\n"
            f"Maneuvers detected: [bold green]{len(events)}[/bold green]",
            title="Scan Results",
            box=box.ROUNDED,
        )
    )

    if events:
        table = Table(
            title="Detected Maneuvers",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("Date", style="cyan")
        table.add_column("Type", style="bold")
        table.add_column("Δ Alt (km)", justify="right")
        table.add_column("Δv (m/s)", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Confidence", justify="right")

        for e in events:
            color = "green" if e.delta_altitude_km > 0 else "red"
            table.add_row(
                f"{e.epoch_after:%Y-%m-%d %H:%M}",
                e.maneuver_type.name.replace("_", " "),
                f"[{color}]{e.delta_altitude_km:+.3f}[/{color}]",
                f"{e.estimated_dv_ms:.2f}",
                f"{e.score_total:.2f}",
                f"{e.confidence:.2f}",
            )

        console.print(table)


def _display_maneuver_table(df):
    """Display a DataFrame of maneuver detections as a rich table."""
    table = Table(box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("Date", style="cyan")
    table.add_column("NORAD", justify="right")
    table.add_column("Name")
    table.add_column("Type", style="bold")
    table.add_column("Δ Alt (km)", justify="right")
    table.add_column("Δv (m/s)", justify="right")
    table.add_column("Score", justify="right")

    for _, row in df.head(50).iterrows():
        epoch = row["epoch_after"]
        if isinstance(epoch, str):
            date_str = epoch[:16]
        else:
            date_str = f"{epoch:%Y-%m-%d %H:%M}"

        delta = row["delta_alt_km"]
        color = "green" if delta > 0 else "red"
        table.add_row(
            date_str,
            str(row["norad_id"]),
            str(row.get("name", "")),
            row["type"].replace("_", " "),
            f"[{color}]{delta:+.3f}[/{color}]",
            f"{row['estimated_dv_ms']:.2f}",
            f"{row['score']:.2f}",
        )

    if len(df) > 50:
        console.print(f"(showing 50 of {len(df)} events)")
    console.print(table)


if __name__ == "__main__":
    main()
