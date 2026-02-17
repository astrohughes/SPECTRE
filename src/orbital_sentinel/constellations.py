"""
Known constellation metadata and NORAD ID lists.

These are periodically updated. For the latest catalog, query Space-Track
directly with `orbital-sentinel constellation --name starlink`.

This file provides a starting point and example IDs for each constellation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .detector import DetectionThresholds


@dataclass
class ConstellationInfo:
    """Metadata for a known satellite constellation."""
    name: str
    operator: str
    altitude_km: float
    inclination_deg: float
    num_planes: int
    sats_per_plane: int
    description: str
    thresholds: DetectionThresholds
    example_norad_ids: list[int]  # A few example IDs for quick testing
    spacetrack_name_pattern: str  # Pattern for Space-Track name search


# ── Known constellations ──

STARLINK = ConstellationInfo(
    name="Starlink",
    operator="SpaceX",
    altitude_km=550,
    inclination_deg=53.0,
    num_planes=72,
    sats_per_plane=22,
    description="SpaceX broadband internet constellation. Multiple shells at 53°, 70°, 97.6°.",
    thresholds=DetectionThresholds.for_starlink(),
    example_norad_ids=[
        44713, 44714, 44715, 44716, 44717,  # v1.0 batch
        48601, 48602, 48603, 48604, 48605,  # later batch
    ],
    spacetrack_name_pattern="STARLINK",
)

ONEWEB = ConstellationInfo(
    name="OneWeb",
    operator="Eutelsat OneWeb",
    altitude_km=1200,
    inclination_deg=87.9,
    num_planes=12,
    sats_per_plane=49,
    description="OneWeb broadband LEO constellation at 1200 km polar orbit.",
    thresholds=DetectionThresholds.for_oneweb(),
    example_norad_ids=[
        44057, 44058, 44059, 44060, 44061,  # early batch
        47694, 47695, 47696, 47697, 47698,  # later batch
    ],
    spacetrack_name_pattern="ONEWEB",
)

IRIDIUM_NEXT = ConstellationInfo(
    name="Iridium NEXT",
    operator="Iridium Communications",
    altitude_km=780,
    inclination_deg=86.4,
    num_planes=6,
    sats_per_plane=11,
    description="Iridium NEXT voice/data constellation at 780 km near-polar orbit.",
    thresholds=DetectionThresholds.for_iridium(),
    example_norad_ids=[
        43039, 43040, 43041, 43042, 43043,  # Iridium NEXT batch
        43075, 43076, 43077, 43078, 43079,
    ],
    spacetrack_name_pattern="IRIDIUM",
)

PLANET_FLOCK = ConstellationInfo(
    name="Planet Flock",
    operator="Planet Labs",
    altitude_km=475,
    inclination_deg=97.4,
    num_planes=1,
    sats_per_plane=120,
    description="Planet Labs Dove imaging constellation in sun-synchronous orbit.",
    thresholds=DetectionThresholds(
        sma_jump_km=0.2,        # High-drag, frequent natural decay
        sma_maintenance_km=0.05,
        inclination_jump_deg=0.005,
    ),
    example_norad_ids=[
        43783, 43784, 43785, 43786, 43787,
    ],
    spacetrack_name_pattern="FLOCK",
)

# Registry
CONSTELLATIONS: dict[str, ConstellationInfo] = {
    "starlink": STARLINK,
    "oneweb": ONEWEB,
    "iridium": IRIDIUM_NEXT,
    "planet": PLANET_FLOCK,
}


def get_constellation(name: str) -> Optional[ConstellationInfo]:
    """Look up a constellation by name (case-insensitive)."""
    return CONSTELLATIONS.get(name.lower())


def list_constellations() -> list[str]:
    """List available constellation presets."""
    return list(CONSTELLATIONS.keys())
