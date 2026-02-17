"""
Example: Maneuver detection on synthetic TLE data.

This example doesn't require Space-Track credentials — it generates
synthetic TLEs with known maneuvers and runs detection on them.
Useful for understanding how the detector works and tuning thresholds.
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from spectre.tle_parser import TLE
from spectre.detector import (
    ManeuverDetector,
    DetectionThresholds,
    build_element_history,
    detect_maneuvers_batch,
)


def make_synthetic_tle(
    norad_id: int,
    epoch: datetime,
    mean_motion: float = 15.058,
    inclination: float = 53.0,
    raan: float = 0.0,
    eccentricity: float = 0.00015,
    name: str = "SYNTH-SAT",
) -> TLE:
    """Create a synthetic TLE object."""
    epoch_day = (epoch - datetime(epoch.year, 1, 1)).total_seconds() / 86400.0 + 1.0
    return TLE(
        name=name,
        norad_id=norad_id,
        intl_designator="24001A",
        classification="U",
        epoch_year=epoch.year,
        epoch_day=epoch_day,
        epoch_dt=epoch,
        mean_motion_dot=0.0000012,
        mean_motion_ddot=0.0,
        bstar=6.2e-5,
        inclination=inclination,
        raan=raan,
        eccentricity=eccentricity,
        arg_perigee=90.0,
        mean_anomaly=0.0,
        mean_motion=mean_motion,
        rev_number=1000,
    )


def main():
    print("=" * 65)
    print("  SPECTRE — Synthetic Maneuver Detection Demo")
    print("=" * 65)

    base = datetime(2024, 6, 1)
    tles = {}

    # ── Satellite A: Altitude raise on day 15 ──
    sat_a = []
    for day in range(30):
        epoch = base + timedelta(days=day, hours=6)
        if day < 15:
            mm = 15.058 - day * 0.00001  # Tiny natural drag
        else:
            mm = 15.048 - (day - 15) * 0.00001  # Jump! Then resume drift
        sat_a.append(make_synthetic_tle(55001, epoch, mm, name="DEMO-A"))
    tles[55001] = sat_a

    # ── Satellite B: Inclination change on day 10 ──
    sat_b = []
    for day in range(30):
        epoch = base + timedelta(days=day, hours=6)
        inc = 53.0 if day < 10 else 53.05
        sat_b.append(make_synthetic_tle(55002, epoch, inclination=inc, name="DEMO-B"))
    tles[55002] = sat_b

    # ── Satellite C: Quiet — no maneuvers ──
    sat_c = []
    for day in range(30):
        epoch = base + timedelta(days=day, hours=6)
        mm = 15.058 - day * 0.00001
        sat_c.append(make_synthetic_tle(55003, epoch, mm, name="DEMO-C"))
    tles[55003] = sat_c

    # ── Satellite D: Multiple small drag makeup burns ──
    sat_d = []
    mm = 15.058
    for day in range(30):
        epoch = base + timedelta(days=day, hours=6)
        mm -= 0.00002  # Gradual drag decay
        if day in (7, 14, 21):  # Periodic correction
            mm += 0.008  # Small boost
        sat_d.append(make_synthetic_tle(55004, epoch, mm, name="DEMO-D"))
    tles[55004] = sat_d

    # ── Run detection ──
    print(f"\nGenerated {sum(len(v) for v in tles.values())} TLEs for {len(tles)} satellites")
    print()

    df = detect_maneuvers_batch(tles)

    if df.empty:
        print("No maneuvers detected.")
        return

    print(f"Detected {len(df)} maneuver events:\n")
    print(f"{'DATE':20s} {'NORAD':>6} {'NAME':10s} {'TYPE':25s} {'Δ ALT (km)':>11} {'Δv (m/s)':>9} {'SCORE':>6}")
    print("-" * 90)

    for _, row in df.iterrows():
        epoch = row["epoch_after"]
        if isinstance(epoch, str):
            date_str = epoch[:19]
        else:
            date_str = f"{epoch:%Y-%m-%d %H:%M}"

        delta = row["delta_alt_km"]
        arrow = "↑" if delta > 0 else "↓"

        print(
            f"{date_str:20s} "
            f"{row['norad_id']:>6} "
            f"{row.get('name', ''):10s} "
            f"{row['type']:25s} "
            f"{arrow}{abs(delta):>10.3f} "
            f"{row['estimated_dv_ms']:>9.2f} "
            f"{row['score']:>6.2f}"
        )

    # ── Per-satellite summary ──
    print(f"\n{'=' * 65}")
    print("PER-SATELLITE SUMMARY")
    print(f"{'=' * 65}")

    for norad_id in sorted(tles.keys()):
        sat_events = df[df["norad_id"] == norad_id]
        sat_name = tles[norad_id][0].name

        if sat_events.empty:
            status = "QUIET — no maneuvers detected"
        else:
            n = len(sat_events)
            total_dv = sat_events["estimated_dv_ms"].sum()
            types = sat_events["type"].value_counts().to_dict()
            type_str = ", ".join(f"{v}x {k.replace('_', ' ')}" for k, v in types.items())
            status = f"{n} maneuver(s), Δv≈{total_dv:.1f} m/s — {type_str}"

        print(f"  NORAD {norad_id} ({sat_name}): {status}")

    # ── Generate plots if matplotlib is available ──
    try:
        from spectre.viz import plot_element_history, plot_maneuver_timeline
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        # Plot satellite A (the one with the altitude raise)
        elem_df = build_element_history(tles[55001])
        sat_a_events = df[df["norad_id"] == 55001]

        fig = plot_element_history(
            elem_df, sat_a_events,
            title="DEMO-A: Altitude Raise Detection",
            save_path="data/demo_element_history.png",
        )
        print(f"\nPlot saved to data/demo_element_history.png")

        fig = plot_maneuver_timeline(
            df, title="Demo — All Detections",
            save_path="data/demo_timeline.png",
        )
        print(f"Plot saved to data/demo_timeline.png")

    except ImportError:
        print("\nInstall matplotlib for visualization: pip install matplotlib")


if __name__ == "__main__":
    main()
