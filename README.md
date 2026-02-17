# 🛰️ Orbital Sentinel

Detect satellite maneuvers from publicly available TLE data — an OSINT tool for analyzing constellation operations.

---

## What this does

Every satellite maneuver leaves a fingerprint in Two-Line Element (TLE) data published by the U.S. Space Force via [Space-Track.org](https://www.space-track.org). Orbital Sentinel detects these fingerprints automatically by analyzing discontinuities in orbital elements across consecutive TLE epochs.

**Detection channels** (fused for robust classification):
- Semi-major axis jumps → altitude change maneuvers
- Mean motion residuals → deviation from expected drag trend
- Inclination discontinuities → plane change maneuvers
- RAAN residuals → unexpected node shift after removing J2 drift
- Eccentricity jumps → orbit shape changes
- B* anomalies → drag coefficient discontinuities

**What you can learn:**
- Which satellites in a constellation are actively maneuvering
- How often different operators perform station-keeping
- Whether a constellation is raising orbits, deorbiting, or repositioning
- Estimated delta-v budgets from TLE-observable effects
- Maneuver cadence and operational patterns

## Quick Start

### Install

```bash
git clone https://github.com/YOUR_USERNAME/orbital-sentinel.git
cd orbital-sentinel
pip install -e ".[dev]"
```

### Run the offline demo (no Space-Track account needed)

```bash
python examples/synthetic_demo.py
```

This generates synthetic TLEs with known maneuvers and shows the detector finding them.

### Scan a real satellite

```bash
# Set Space-Track credentials (free account)
export SPACETRACK_USER="your@email.com"
export SPACETRACK_PASS="your_password"

# Scan ISS for maneuvers in the last 90 days
orbital-sentinel scan --norad-id 25544 --days 90

# Scan from a local TLE file
orbital-sentinel scan --file data/my_tles.txt
```

### Scan a constellation

```bash
# Starlink example (uses built-in NORAD ID presets)
orbital-sentinel constellation --name starlink \
    --norad-ids 44713,44714,44715 \
    --days 30 \
    --output starlink_maneuvers.csv \
    --report-dir data/reports/starlink
```

### Python API

```python
from orbital_sentinel.spacetrack import SpaceTrackClient
from orbital_sentinel.detector import ManeuverDetector, DetectionThresholds
from orbital_sentinel.viz import plot_element_history

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
orbital-sentinel/
├── src/orbital_sentinel/
│   ├── __init__.py
│   ├── tle_parser.py       # TLE parsing with derived quantities
│   ├── spacetrack.py       # Space-Track API client with caching
│   ├── detector.py         # Core maneuver detection engine
│   ├── constellations.py   # Known constellation presets
│   ├── viz.py              # Matplotlib visualization tools
│   └── cli.py              # Click CLI interface
├── tests/
│   └── test_orbital_sentinel.py
├── examples/
│   ├── synthetic_demo.py   # Offline demo with synthetic data
│   └── starlink_scan.py    # Real Starlink analysis
└── data/                   # Cache, reports, exports
```

## Detection Method

For each consecutive TLE pair (t₁, t₂):

1. **Compute element differences** — Δa, Δi, Δe, Δn, ΔRAAN
2. **Subtract expected drift** — Remove J2 secular RAAN regression and drag-induced SMA decay (estimated from ṅ in the TLE)
3. **Score each channel** — Normalize residuals against configurable thresholds to get per-channel suspicion scores (0–1)
4. **Fuse scores** — Weighted sum across all channels produces a total maneuver score
5. **Classify** — Based on which channels fired strongest: altitude raise/lower/maintenance, plane change, phasing, orbit raise, or deorbit
6. **Merge** — Closely-spaced detections (within 6 hours) are merged, keeping the highest-scoring event

### Tuning

Presets are included for major constellations. For custom orbits:

```python
thresholds = DetectionThresholds(
    sma_jump_km=0.1,           # SMA discontinuity trigger
    inclination_jump_deg=0.005, # Inclination change trigger
    mean_motion_jump=0.005,     # Mean motion residual trigger
    maneuver_score_threshold=0.3,  # Minimum fused score
)
```

## Constellation Presets

| Constellation | Altitude | Inclination | Key behaviors |
|---------------|----------|-------------|---------------|
| **Starlink** | 550 km | 53° | Frequent orbit raising after deploy, periodic drag makeup |
| **OneWeb** | 1200 km | 87.9° | Minimal drag, occasional plane adjustments |
| **Iridium NEXT** | 780 km | 86.4° | Stable, infrequent maneuvers |
| **Planet Flock** | 475 km | 97.4° | High drag, short-lived, frequent corrections |

## Limitations

- **TLE precision** — TLEs are mean elements with ~1 km position accuracy. Very small maneuvers (< 0.01 m/s) may be undetectable.
- **TLE update cadence** — Space-Track updates TLEs every few hours to days. Fast maneuver sequences may appear as a single event.
- **Drag modeling** — The exponential atmosphere model is a rough proxy. Solar activity variations can mimic small maneuvers.
- **Element type mixing** — TLEs use Brouwer mean elements; direct comparison across epochs assumes consistent fitting.

## License

MIT

## See also

- [SCARAB](https://github.com/YOUR_USERNAME/scarab) — Companion project for constellation station-keeping planning (Rust + Python)
- [Space-Track.org](https://www.space-track.org) — Free TLE data source
- [CelesTrak](https://celestrak.org) — Curated TLE catalog and supplemental data
