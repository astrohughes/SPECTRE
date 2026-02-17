# 👻 SPECTRE

**S**pacecraft **P**ropulsive **E**vent **C**lassification & **T**racking from **R**epeated **E**lements

Detect orbital maneuvers from publicly available TLE data — an Open-Source Intelligence (OSINT) tool for analyzing spacecraft constellation operations.

**Authors:** Kyle Hughes ([@astrohughes](https://github.com/astrohughes/SPECTRE)) — kyle.evan.hughes@gmail.com

---

## What this does

Every spacecraft maneuver leaves a fingerprint in Two-Line Element (TLE) data published by the U.S. Space Force via [Space-Track.org](https://www.space-track.org). SPECTRE detects these fingerprints automatically by analyzing discontinuities in orbital elements across consecutive TLE epochs.

**Detection channels** (fused for robust classification):

- Semi-major axis jumps → altitude change maneuvers
- Mean motion residuals → deviation from expected drag trend
- Inclination discontinuities → plane change maneuvers
- RAAN residuals → unexpected node shift after removing J2 drift
- Eccentricity jumps → orbit shape changes
- B* anomalies → drag coefficient discontinuities

**What you can learn:**

- Which spacecraft in a constellation are actively maneuvering
- How often different operators perform station-keeping
- Whether a constellation is raising orbits, deorbiting, or repositioning
- Estimated delta-v budgets from TLE-observable effects
- Maneuver cadence and operational patterns

## Quick Start

### Install

```bash
git clone https://github.com/huqhesy/spectre.git
cd spectre
pip install -e ".[dev]"
```

### Run the offline demo (no Space-Track account needed)

```bash
python examples/synthetic_demo.py
```

Generates synthetic TLEs with known maneuvers and shows SPECTRE finding them.

### Scan a real spacecraft

```bash
# Set Space-Track credentials (free account)
export SPACETRACK_USER="your@email.com"
export SPACETRACK_PASS="your_password"

# Scan ISS for maneuvers in the last 90 days
spectre scan --norad-id 25544 --days 90

# Scan from a local TLE file
spectre scan --file data/my_tles.txt
```

### Scan a constellation

```bash
spectre constellation --name starlink \
    --norad-ids 44713,44714,44715 \
    --days 30 \
    --output starlink_maneuvers.csv \
    --report-dir data/reports/starlink
```

### Python API

```python
from spectre.spacetrack import SpaceTrackClient
from spectre.detector import ManeuverDetector, DetectionThresholds

# Fetch TLE history
client = SpaceTrackClient()
tles = client.get_tle_history(norad_id=25544, days=90)

# Detect maneuvers
detector = ManeuverDetector(DetectionThresholds.for_starlink())
events = detector.detect(tles)

for event in events:
    print(event.summary())
    # [2024-06-15 14:30] NORAD 25544 (ISS) — ALTITUDE_RAISE ↑0.52 km (Δv≈1.2 m/s, score=0.87)
```

## Architecture

```
spectre/
├── src/spectre/
│   ├── __init__.py          # Package metadata
│   ├── tle_parser.py        # TLE parsing with derived orbital quantities
│   ├── detector.py          # Core maneuver detection engine
│   ├── spacetrack.py        # Space-Track API client with caching
│   ├── constellations.py    # Known constellation presets
│   ├── viz.py               # Matplotlib visualization tools
│   └── cli.py               # Click CLI interface
├── tests/
│   └── test_spectre.py      # Unit and integration tests
├── examples/
│   ├── synthetic_demo.py    # Offline demo with synthetic data
│   └── starlink_scan.py     # Real Starlink analysis
└── data/                    # Cache, reports, exports
```

## Detection Method

For each consecutive TLE pair (t₁, t₂):

1. **Compute element differences** — Δa, Δi, Δe, Δn, ΔRAAN
2. **Subtract expected drift** — Remove J2 secular RAAN regression and drag-induced SMA decay (estimated from ṅ in the TLE)
3. **Score each channel** — Normalize residuals against configurable thresholds to produce per-channel suspicion scores [0, 1]
4. **Fuse scores** — Weighted sum across all channels produces a total maneuver score
5. **Classify** — Based on which channels fired strongest: altitude raise/lower/maintenance, plane change, phasing, orbit raise, or deorbit
6. **Merge** — Closely-spaced detections (< 6 hours) are merged, keeping the highest-scoring event

### Tuning

Presets are included for major constellations. For custom orbits:

```python
thresholds = DetectionThresholds(
    sma_jump_km=0.1,
    inclination_jump_deg=0.005,
    mean_motion_jump=0.005,
    maneuver_score_threshold=0.3,
)
```

## Constellation Presets

| Constellation          | Altitude | Inclination | Characteristics                                           |
| ---------------------- | -------- | ----------- | --------------------------------------------------------- |
| **Starlink**     | 550 km   | 53°        | Frequent orbit raising after deploy, periodic drag makeup |
| **OneWeb**       | 1200 km  | 87.9°      | Minimal drag, occasional plane adjustments                |
| **Iridium NEXT** | 780 km   | 86.4°      | Stable, infrequent maneuvers                              |
| **Planet Flock** | 475 km   | 97.4°      | High drag, short-lived, frequent corrections              |

## Limitations

- **TLE precision** — TLEs are mean elements with ~1 km position accuracy. Very small maneuvers (< 0.01 m/s) may be undetectable.
- **TLE update cadence** — Space-Track updates TLEs every few hours to days. Fast maneuver sequences may appear as a single event.
- **Drag modeling** — The exponential atmosphere proxy is approximate. Solar activity variations can mimic small maneuvers.
- **Element type mixing** — TLEs use Brouwer mean elements; direct comparison across epochs assumes consistent fitting.

## See Also

- [SCARAB](https://github.com/huqhesy/scarab) — Companion project for constellation station-keeping planning (Rust + Python)
- [Space-Track.org](https://www.space-track.org) — Free TLE data source (requires registration)
- [CelesTrak](https://celestrak.org) — Curated TLE catalog and supplemental data

## License

MIT
