"""
Maneuver detection engine.

Detects orbital maneuvers by analyzing discontinuities in TLE-derived
orbital elements across consecutive epochs. Uses multiple detection
methods and fuses results for robust classification.

Detection methods:
1. SMA jump — sudden change in semi-major axis (altitude change maneuver)
2. Mean motion residual — deviation from expected secular trend
3. Inclination change — plane change maneuver
4. RAAN discontinuity — unexpected node shift (plane change or phasing)
5. Eccentricity jump — orbit shape change
6. B* anomaly — drag coefficient discontinuity (often correlated with maneuvers)

Each method produces a "suspicion score" (0-1). Scores are fused using
configurable weights to produce a final maneuver probability.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd

from .tle_parser import TLE

logger = logging.getLogger(__name__)


class ManeuverType(Enum):
    """Classified maneuver types."""
    ALTITUDE_RAISE = auto()     # Prograde burn → higher orbit
    ALTITUDE_LOWER = auto()     # Retrograde burn → lower orbit
    ALTITUDE_MAINTENANCE = auto()  # Small drag makeup
    PLANE_CHANGE = auto()       # Inclination or RAAN correction
    PHASING = auto()            # In-track repositioning
    ORBIT_RAISE = auto()        # Large altitude increase (e.g., orbit raising after deployment)
    DEORBIT = auto()            # Intentional reentry
    UNKNOWN = auto()


@dataclass
class DetectionThresholds:
    """
    Configurable thresholds for maneuver detection.

    Default values tuned for LEO constellations (Starlink, OneWeb, Iridium).
    Adjust for different orbit regimes or satellite types.
    """

    # SMA jump threshold (km). Anything above this between consecutive TLEs
    # is flagged. Natural drag at 550km is ~1-5 m/day, so jumps > 0.05 km
    # over typical TLE intervals (hours to days) are suspicious.
    sma_jump_km: float = 0.1

    # Small SMA change threshold for maintenance maneuvers (km).
    # Catches subtle drag makeup burns.
    sma_maintenance_km: float = 0.03

    # Inclination change threshold (degrees).
    # J2 causes NO secular inclination drift, so any change is from
    # short-period oscillations (~0.001°) or a real maneuver.
    inclination_jump_deg: float = 0.003

    # RAAN discontinuity threshold (degrees).
    # After subtracting expected J2 secular drift.
    raan_residual_deg: float = 0.02

    # Eccentricity jump threshold.
    eccentricity_jump: float = 5e-4

    # Mean motion jump threshold (rev/day).
    mean_motion_jump: float = 0.005

    # B* anomaly threshold (relative change).
    bstar_relative_change: float = 0.5

    # Minimum time between TLEs to consider (hours).
    # Very close TLEs can have noisy differences.
    min_epoch_gap_hours: float = 0.5

    # Maximum time between TLEs (days).
    # Gaps > this make detection unreliable.
    max_epoch_gap_days: float = 5.0

    # Score fusion weights.
    weight_sma: float = 0.30
    weight_mean_motion: float = 0.20
    weight_inclination: float = 0.20
    weight_raan: float = 0.10
    weight_eccentricity: float = 0.10
    weight_bstar: float = 0.10

    # Final score threshold for maneuver declaration.
    maneuver_score_threshold: float = 0.3

    @classmethod
    def for_starlink(cls) -> DetectionThresholds:
        """Thresholds tuned for Starlink (~550 km, high drag environment)."""
        return cls(
            sma_jump_km=0.15,
            sma_maintenance_km=0.04,
            inclination_jump_deg=0.005,
            mean_motion_jump=0.006,
        )

    @classmethod
    def for_oneweb(cls) -> DetectionThresholds:
        """Thresholds for OneWeb (~1200 km, low drag)."""
        return cls(
            sma_jump_km=0.08,
            sma_maintenance_km=0.02,
            inclination_jump_deg=0.003,
            mean_motion_jump=0.003,
        )

    @classmethod
    def for_iridium(cls) -> DetectionThresholds:
        """Thresholds for Iridium NEXT (~780 km)."""
        return cls(
            sma_jump_km=0.10,
            sma_maintenance_km=0.03,
            inclination_jump_deg=0.004,
            mean_motion_jump=0.004,
        )


@dataclass
class ManeuverEvent:
    """A detected maneuver event."""

    norad_id: int
    satellite_name: Optional[str]

    # Timing
    epoch_before: datetime
    epoch_after: datetime
    epoch_gap_hours: float

    # Detection scores (0-1 each)
    score_sma: float
    score_mean_motion: float
    score_inclination: float
    score_raan: float
    score_eccentricity: float
    score_bstar: float
    score_total: float

    # Element changes
    delta_sma_km: float
    delta_altitude_km: float
    delta_inclination_deg: float
    delta_raan_deg: float
    delta_raan_residual_deg: float
    delta_eccentricity: float
    delta_mean_motion: float

    # Estimated delta-v (very rough, from vis-viva)
    estimated_dv_ms: float

    # Classification
    maneuver_type: ManeuverType
    confidence: float  # 0-1

    # Pre/post orbital elements for context
    sma_before_km: float
    sma_after_km: float
    altitude_before_km: float
    altitude_after_km: float
    inclination_before_deg: float
    inclination_after_deg: float

    def to_dict(self) -> dict:
        return {
            "norad_id": self.norad_id,
            "name": self.satellite_name,
            "epoch_before": self.epoch_before,
            "epoch_after": self.epoch_after,
            "gap_hours": round(self.epoch_gap_hours, 2),
            "score": round(self.score_total, 3),
            "confidence": round(self.confidence, 3),
            "type": self.maneuver_type.name,
            "delta_sma_km": round(self.delta_sma_km, 4),
            "delta_alt_km": round(self.delta_altitude_km, 4),
            "delta_inc_deg": round(self.delta_inclination_deg, 5),
            "delta_raan_deg": round(self.delta_raan_deg, 5),
            "estimated_dv_ms": round(self.estimated_dv_ms, 3),
            "alt_before_km": round(self.altitude_before_km, 2),
            "alt_after_km": round(self.altitude_after_km, 2),
        }

    def summary(self) -> str:
        direction = "↑" if self.delta_altitude_km > 0 else "↓"
        return (
            f"[{self.epoch_after:%Y-%m-%d %H:%M}] "
            f"NORAD {self.norad_id} "
            f"({self.satellite_name or 'UNKNOWN'}) — "
            f"{self.maneuver_type.name} "
            f"{direction}{abs(self.delta_altitude_km):.2f} km "
            f"(Δv≈{self.estimated_dv_ms:.1f} m/s, "
            f"score={self.score_total:.2f})"
        )


class ManeuverDetector:
    """
    Core maneuver detection engine.

    Analyzes a time series of TLEs for a single satellite and identifies
    epochs where maneuvers likely occurred.
    """

    def __init__(self, thresholds: Optional[DetectionThresholds] = None):
        self.thresholds = thresholds or DetectionThresholds()

    def detect(self, tles: list[TLE]) -> list[ManeuverEvent]:
        """
        Detect maneuvers from a TLE history for one satellite.

        Args:
            tles: List of TLEs for a single satellite, sorted by epoch.

        Returns:
            List of detected ManeuverEvents, sorted by epoch.
        """
        if len(tles) < 2:
            return []

        # Sort by epoch (should already be, but be safe)
        tles = sorted(tles, key=lambda t: t.epoch_dt)

        # Compute element time series for trend estimation
        epochs = np.array([(t.epoch_dt - tles[0].epoch_dt).total_seconds() for t in tles])
        sma_series = np.array([t.semi_major_axis for t in tles])
        mm_series = np.array([t.mean_motion for t in tles])

        events = []

        for i in range(1, len(tles)):
            prev = tles[i - 1]
            curr = tles[i]

            # Check epoch gap
            gap = (curr.epoch_dt - prev.epoch_dt).total_seconds() / 3600.0
            if gap < self.thresholds.min_epoch_gap_hours:
                continue
            if gap > self.thresholds.max_epoch_gap_days * 24.0:
                continue

            # Compute scores for each detection channel
            scores = self._compute_scores(prev, curr, gap)

            # Fuse scores
            t = self.thresholds
            total = (
                t.weight_sma * scores["sma"]
                + t.weight_mean_motion * scores["mean_motion"]
                + t.weight_inclination * scores["inclination"]
                + t.weight_raan * scores["raan"]
                + t.weight_eccentricity * scores["eccentricity"]
                + t.weight_bstar * scores["bstar"]
            )

            if total >= t.maneuver_score_threshold:
                event = self._build_event(prev, curr, gap, scores, total)
                events.append(event)

        # Merge closely-spaced detections (within 6 hours)
        events = self._merge_nearby(events)

        return events

    def _compute_scores(self, prev: TLE, curr: TLE, gap_hours: float) -> dict[str, float]:
        """Compute suspicion scores for each detection channel."""
        t = self.thresholds

        # Normalize by epoch gap — larger gaps mean more natural drift
        gap_days = gap_hours / 24.0

        # ── SMA jump ──
        delta_sma = curr.semi_major_axis - prev.semi_major_axis
        # Expected SMA change from drag (very rough: use mean_motion_dot)
        expected_sma_change = self._expected_sma_drift(prev, gap_days)
        sma_residual = abs(delta_sma - expected_sma_change)
        score_sma = min(1.0, sma_residual / t.sma_jump_km)

        # ── Mean motion ──
        delta_mm = abs(curr.mean_motion - prev.mean_motion)
        expected_mm_change = abs(prev.mean_motion_dot * 2.0 * gap_days)
        mm_residual = abs(delta_mm - expected_mm_change)
        score_mm = min(1.0, mm_residual / t.mean_motion_jump)

        # ── Inclination ──
        delta_inc = abs(curr.inclination - prev.inclination)
        # J2 doesn't cause secular inclination drift, so any change is real
        score_inc = min(1.0, delta_inc / t.inclination_jump_deg)

        # ── RAAN ──
        delta_raan = _angle_diff(curr.raan, prev.raan)
        expected_raan = prev.raan_rate * gap_days
        raan_residual = abs(delta_raan - expected_raan)
        # Wrap to [-180, 180]
        if raan_residual > 180:
            raan_residual = 360 - raan_residual
        score_raan = min(1.0, raan_residual / t.raan_residual_deg)

        # ── Eccentricity ──
        delta_ecc = abs(curr.eccentricity - prev.eccentricity)
        score_ecc = min(1.0, delta_ecc / t.eccentricity_jump)

        # ── B* anomaly ──
        if abs(prev.bstar) > 1e-10:
            bstar_change = abs(curr.bstar - prev.bstar) / abs(prev.bstar)
        else:
            bstar_change = 0.0
        score_bstar = min(1.0, bstar_change / t.bstar_relative_change)

        return {
            "sma": score_sma,
            "mean_motion": score_mm,
            "inclination": score_inc,
            "raan": score_raan,
            "eccentricity": score_ecc,
            "bstar": score_bstar,
        }

    def _expected_sma_drift(self, tle: TLE, gap_days: float) -> float:
        """
        Estimate expected SMA change from drag over a time interval.

        Uses the TLE's mean motion derivative as a proxy for drag.
        Δn ≈ 2 * ṅ₁ * Δt → Δa ≈ -2a/(3n) * Δn
        """
        n = tle.mean_motion * 2 * math.pi / 86400.0  # rad/s
        n_dot = tle.mean_motion_dot * 2.0  # The TLE field is ṅ/2
        delta_n = n_dot * (2 * math.pi / 86400.0) * gap_days  # rad/s change

        if abs(n) < 1e-10:
            return 0.0

        # Δa = -2a/(3n) * Δn
        return -2.0 * tle.semi_major_axis / (3.0 * n) * delta_n

    def _build_event(
        self,
        prev: TLE,
        curr: TLE,
        gap_hours: float,
        scores: dict[str, float],
        total_score: float,
    ) -> ManeuverEvent:
        """Build a ManeuverEvent from detection results."""
        delta_sma = curr.semi_major_axis - prev.semi_major_axis
        delta_alt = curr.altitude - prev.altitude
        delta_inc = curr.inclination - prev.inclination
        delta_raan = _angle_diff(curr.raan, prev.raan)
        delta_ecc = curr.eccentricity - prev.eccentricity
        delta_mm = curr.mean_motion - prev.mean_motion

        # Expected RAAN drift
        expected_raan = prev.raan_rate * (gap_hours / 24.0)
        raan_residual = delta_raan - expected_raan

        # Estimate delta-v from vis-viva (rough)
        estimated_dv = self._estimate_delta_v(prev, curr)

        # Classify
        maneuver_type = self._classify(delta_alt, delta_inc, delta_ecc, scores)

        # Confidence based on score and gap quality
        gap_quality = 1.0 - min(1.0, gap_hours / (self.thresholds.max_epoch_gap_days * 24))
        confidence = min(1.0, total_score * (0.5 + 0.5 * gap_quality))

        return ManeuverEvent(
            norad_id=curr.norad_id,
            satellite_name=curr.name or prev.name,
            epoch_before=prev.epoch_dt,
            epoch_after=curr.epoch_dt,
            epoch_gap_hours=gap_hours,
            score_sma=scores["sma"],
            score_mean_motion=scores["mean_motion"],
            score_inclination=scores["inclination"],
            score_raan=scores["raan"],
            score_eccentricity=scores["eccentricity"],
            score_bstar=scores["bstar"],
            score_total=total_score,
            delta_sma_km=delta_sma,
            delta_altitude_km=delta_alt,
            delta_inclination_deg=delta_inc,
            delta_raan_deg=delta_raan,
            delta_raan_residual_deg=raan_residual,
            delta_eccentricity=delta_ecc,
            delta_mean_motion=delta_mm,
            estimated_dv_ms=estimated_dv,
            maneuver_type=maneuver_type,
            confidence=confidence,
            sma_before_km=prev.semi_major_axis,
            sma_after_km=curr.semi_major_axis,
            altitude_before_km=prev.altitude,
            altitude_after_km=curr.altitude,
            inclination_before_deg=prev.inclination,
            inclination_after_deg=curr.inclination,
        )

    def _estimate_delta_v(self, prev: TLE, curr: TLE) -> float:
        """
        Rough delta-v estimate from vis-viva equation (m/s).

        For small SMA changes: Δv ≈ (v/2) * |Δa/a|
        For inclination changes: Δv ≈ v * |Δi| (radians)
        """
        MU = 398600.4418
        a = prev.semi_major_axis
        v = math.sqrt(MU / a) * 1000  # m/s

        # In-plane component
        delta_a = abs(curr.semi_major_axis - prev.semi_major_axis)
        dv_inplane = v / 2.0 * delta_a / a if a > 0 else 0.0

        # Out-of-plane component
        delta_i_rad = math.radians(abs(curr.inclination - prev.inclination))
        dv_outplane = v * delta_i_rad

        return math.sqrt(dv_inplane**2 + dv_outplane**2)

    def _classify(
        self,
        delta_alt: float,
        delta_inc: float,
        delta_ecc: float,
        scores: dict[str, float],
    ) -> ManeuverType:
        """Classify the maneuver type based on which channels fired."""
        t = self.thresholds

        # Large altitude changes
        if abs(delta_alt) > 50:
            if delta_alt > 0:
                return ManeuverType.ORBIT_RAISE
            else:
                return ManeuverType.DEORBIT

        # Plane change dominant
        if scores["inclination"] > 0.5 or scores["raan"] > 0.5:
            if scores["sma"] < 0.3:
                return ManeuverType.PLANE_CHANGE

        # Altitude maneuvers
        if scores["sma"] > 0.3:
            if abs(delta_alt) < t.sma_maintenance_km * 2:
                return ManeuverType.ALTITUDE_MAINTENANCE
            elif delta_alt > 0:
                return ManeuverType.ALTITUDE_RAISE
            else:
                return ManeuverType.ALTITUDE_LOWER

        # Phasing (mean motion change without proportional SMA change)
        if scores["mean_motion"] > 0.4 and scores["sma"] < 0.3:
            return ManeuverType.PHASING

        return ManeuverType.UNKNOWN

    def _merge_nearby(
        self, events: list[ManeuverEvent], window_hours: float = 6.0
    ) -> list[ManeuverEvent]:
        """Merge detections within a time window (keep highest-scoring)."""
        if not events:
            return events

        merged = [events[0]]
        for event in events[1:]:
            prev = merged[-1]
            gap = (event.epoch_after - prev.epoch_after).total_seconds() / 3600.0

            if gap < window_hours:
                # Keep the one with higher score
                if event.score_total > prev.score_total:
                    merged[-1] = event
            else:
                merged.append(event)

        return merged


def detect_maneuvers_batch(
    tle_dict: dict[int, list[TLE]],
    thresholds: Optional[DetectionThresholds] = None,
) -> pd.DataFrame:
    """
    Detect maneuvers across an entire constellation.

    Args:
        tle_dict: Dict mapping NORAD ID → list of TLEs
        thresholds: Detection thresholds (default: standard LEO)

    Returns:
        DataFrame with all detected maneuver events.
    """
    detector = ManeuverDetector(thresholds)
    all_events = []

    for norad_id, tles in tle_dict.items():
        events = detector.detect(tles)
        for event in events:
            all_events.append(event.to_dict())

    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    df = df.sort_values("epoch_after").reset_index(drop=True)
    return df


def build_element_history(tles: list[TLE]) -> pd.DataFrame:
    """
    Convert a TLE list to a DataFrame of orbital elements over time.

    Useful for plotting element evolution and visually confirming detections.
    """
    records = [tle.to_dict() for tle in tles]
    df = pd.DataFrame(records)
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, wrapped to [-180, 180]."""
    d = (a - b) % 360
    if d > 180:
        d -= 360
    return d
