#!/usr/bin/env python3
"""
SPECTRE Example: Detect Starlink maneuvers from Space-Track data.

Requires Space-Track credentials:
    export SPACETRACK_USER="your@email.com"
    export SPACETRACK_PASS="your_password"

Register free at: https://www.space-track.org/auth/createAccount
"""
import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from spectre.spacetrack import SpaceTrackClient
from spectre.detector import (
    ManeuverDetector,
    DetectionThresholds,
    detect_maneuvers_batch,
    build_element_history,
)
from spectre.constellations import STARLINK
from spectre.viz import (
    plot_element_history,
    plot_maneuver_timeline,
    plot_constellation_activity,
    generate_report,
)


def main():
    print("=" * 65)
    print("  SPECTRE — Starlink Maneuver Detection")
    print("=" * 65)

    client = SpaceTrackClient()

    # Use example NORAD IDs from the Starlink preset
    norad_ids = STARLINK.example_norad_ids
    print(f"\nAnalyzing {len(norad_ids)} Starlink satellites (90-day window)")

    end = datetime.utcnow()
    start = end - timedelta(days=90)

    # Fetch TLE history
    print("\nFetching TLE history from Space-Track...")
    tle_dict = client.get_constellation_tles(norad_ids, start, end)
    total_tles = sum(len(v) for v in tle_dict.values())
    print(f"Fetched {total_tles} TLEs for {len(tle_dict)} satellites")

    # Run detection with Starlink-tuned thresholds
    print("\nRunning maneuver detection...")
    df = detect_maneuvers_batch(tle_dict, STARLINK.thresholds)

    print(f"\nDetected {len(df)} maneuver events")

    if not df.empty:
        # Print summary table
        print(f"\n{'DATE':20s} {'NORAD':>6} {'TYPE':25s} {'Δ ALT':>8} {'Δv':>7} {'SCORE':>6}")
        print("-" * 75)
        for _, row in df.iterrows():
            epoch = row["epoch_after"]
            date_str = f"{epoch:%Y-%m-%d %H:%M}" if hasattr(epoch, "strftime") else str(epoch)[:16]
            delta = row["delta_alt_km"]
            arrow = "↑" if delta > 0 else "↓"
            print(
                f"{date_str:20s} "
                f"{row['norad_id']:>6} "
                f"{row['type']:25s} "
                f"{arrow}{abs(delta):>7.3f} "
                f"{row['estimated_dv_ms']:>7.2f} "
                f"{row['score']:>6.2f}"
            )

        # Generate report
        print("\nGenerating report...")
        report_path = generate_report(
            df, constellation_name="Starlink", output_dir="data/reports/starlink"
        )
        print(f"Report saved to {report_path}")

        # Save raw results
        df.to_csv("data/starlink_maneuvers.csv", index=False)
        print("Results saved to data/starlink_maneuvers.csv")


if __name__ == "__main__":
    main()
