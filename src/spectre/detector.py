#!/usr/bin/env python3
"""Core maneuver detection engine.

Detects orbital maneuvers by analyzing discontinuities in TLE-derived
orbital elements across consecutive epochs. Uses multiple independent
detection channels fused into a single maneuver score.

Detection channels:
    1. SMA jump — sudden semi-major axis change (altitude maneuver).
    2. Mean motion residual — deviation from expected secular drag trend.
    3. Inclination change — plane change maneuver.
    4. RAAN discontinuity — unexpected node shift after removing J2 drift.
    5. Eccentricity jump — orbit shape change.
    6. B* anomaly — drag coefficient discontinuity.

Each channel produces a suspicion score in [0, 1]. Scores are fused via
configurable weights to produce a final maneuver probability.
"""
from __future__ import annotations

import logging
import math
import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from .tle_parser import TLE

logger = logging.getLogger(__name__)


# Enumerations
class ManeuverType(Enum):
    """Classified maneuver types inferred from element discontinuities.
    """
    ALTITUDE_RAISE = auto()
    ALTITUDE_LOWER = auto()
    ALTITUDE_MAINTENANCE = auto()
    PLANE_CHANGE = auto()
    PHASING = auto()
    ORBIT_RAISE = auto()
    DEORBIT = auto()
    UNKNOWN = auto()


# Configuration
@dataclass
class DetectionThresholds:
    """Configurable thresholds for maneuver detection.

    Default values are tuned for LEO constellations (Starlink, OneWeb,
    Iridium). Adjust for different orbit regimes or spacecraft types.

    TODO: Make WAY more configurable.

    Attributes:
        sma_jump_km: SMA discontinuity trigger (km).
        sma_maintenance_km: Small SMA change for drag makeup (km).
        inclination_jump_deg: Inclination change trigger (degrees).
        raan_residual_deg: RAAN residual after J2 removal (degrees).
        eccentricity_jump: Eccentricity discontinuity trigger.
        mean_motion_jump: Mean motion residual trigger (rev/day).
        bstar_relative_change: B* relative change trigger.
        min_epoch_gap_hours: Minimum TLE spacing to consider (hours).
        max_epoch_gap_days: Maximum TLE gap for reliable detection (days).
        weight_sma: Fusion weight for SMA channel.
        weight_mean_motion: Fusion weight for mean motion channel.
        weight_inclination: Fusion weight for inclination channel.
        weight_raan: Fusion weight for RAAN channel.
        weight_eccentricity: Fusion weight for eccentricity channel.
        weight_bstar: Fusion weight for B* channel.
        maneuver_score_threshold: Minimum fused score to declare a maneuver.
    """
    sma_jump_km: float = 0.1
    sma_maintenance_km: float = 0.03
    inclination_jump_deg: float = 0.003
    raan_residual_deg: float = 0.02
    eccentricity_jump: float = 5e-4
    mean_motion_jump: float = 0.005
    bstar_relative_change: float = 0.5
    min_epoch_gap_hours: float = 0.5
    max_epoch_gap_days: float = 5.0
    weight_sma: float = 0.30
    weight_mean_motion: float = 0.20
    weight_inclination: float = 0.20
    weight_raan: float = 0.10
    weight_eccentricity: float = 0.10
    weight_bstar: float = 0.10
    maneuver_score_threshold: float = 0.3

    @classmethod
    def for_starlink(cls) -> DetectionThresholds:
        """Thresholds tuned for Starlink (~550 km, high drag).
        """
        return cls(
            sma_jump_km=0.15,
            sma_maintenance_km=0.04,
            inclination_jump_deg=0.005,
            mean_motion_jump=0.006,
        )

    @classmethod
    def for_oneweb(cls) -> DetectionThresholds:
        """Thresholds tuned for OneWeb (~1200 km, low drag).
        """
        return cls(
            sma_jump_km=0.08,
            sma_maintenance_km=0.02,
            inclination_jump_deg=0.003,
            mean_motion_jump=0.003,
        )

    @classmethod
    def for_iridium(cls) -> DetectionThresholds:
        """Thresholds tuned for Iridium NEXT (~780 km).
        """
        return cls(
            sma_jump_km=0.10,
            sma_maintenance_km=0.03,
            inclination_jump_deg=0.004,
            mean_motion_jump=0.004,
        )


# Maneuver event
@dataclass
class ManeuverEvent:
    """A detected maneuver event with full context.

    Attributes:
        norad_id: NORAD catalog number of the spacecraft.
        spacecraft_name: Spacecraft name (if available).
        epoch_before: TLE epoch immediately before the maneuver.
        epoch_after: TLE epoch immediately after the maneuver.
        epoch_gap_hours: Time between the two TLEs (hours).
        score_sma: SMA channel suspicion score [0, 1].
        score_mean_motion: Mean motion channel score [0, 1].
        score_inclination: Inclination channel score [0, 1].
        score_raan: RAAN channel score [0, 1].
        score_eccentricity: Eccentricity channel score [0, 1].
        score_bstar: B* channel score [0, 1].
        score_total: Fused maneuver score [0, 1].
        delta_sma_km: Change in semi-major axis (km).
        delta_altitude_km: Change in altitude (km).
        delta_inclination_deg: Change in inclination (degrees).
        delta_raan_deg: Raw RAAN change (degrees).
        delta_raan_residual_deg: RAAN residual after J2 removal (degrees).
        delta_eccentricity: Change in eccentricity.
        delta_mean_motion: Change in mean motion (rev/day).
        estimated_dv_ms: Rough delta-v estimate from vis-viva (m/s).
        maneuver_type: Classified maneuver type.
        confidence: Detection confidence [0, 1].
        sma_before_km: Pre-maneuver semi-major axis (km).
        sma_after_km: Post-maneuver semi-major axis (km).
        altitude_before_km: Pre-maneuver altitude (km).
        altitude_after_km: Post-maneuver altitude (km).
        inclination_before_deg: Pre-maneuver inclination (degrees).
        inclination_after_deg: Post-maneuver inclination (degrees).
    """
    norad_id: int
    spacecraft_name: Optional[str]
    epoch_before: datetime
    epoch_after: datetime
    epoch_gap_hours: float
    score_sma: float
    score_mean_motion: float
    score_inclination: float
    score_raan: float
    score_eccentricity: float
    score_bstar: float
    score_total: float
    delta_sma_km: float
    delta_altitude_km: float
    delta_inclination_deg: float
    delta_raan_deg: float
    delta_raan_residual_deg: float
    delta_eccentricity: float
    delta_mean_motion: float
    estimated_dv_ms: float
    maneuver_type: ManeuverType
    confidence: float
    sma_before_km: float
    sma_after_km: float
    altitude_before_km: float
    altitude_after_km: float
    inclination_before_deg: float
    inclination_after_deg: float

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary for DataFrame construction."""
        return {
            "norad_id": self.norad_id,
            "name": self.spacecraft_name,
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
        """Return a one-line human-readable summary of the event."""
        direction = "↑" if self.delta_altitude_km > 0 else "↓"
        return (
            f"[{self.epoch_after:%Y-%m-%d %H:%M}] "
            f"NORAD {self.norad_id} "
            f"({self.spacecraft_name or 'UNKNOWN'}) — "
            f"{self.maneuver_type.name} "
            f"{direction}{abs(self.delta_altitude_km):.2f} km "
            f"(Δv≈{self.estimated_dv_ms:.1f} m/s, "
            f"score={self.score_total:.2f})"
        )


# Detection engine
class ManeuverDetector:
    """Core maneuver detection engine.

    Analyzes a time series of TLEs for a single spacecraft and identifies
    epochs where maneuvers likely occurred by fusing multiple independent
    detection channels.

    Args:
        thresholds: Detection thresholds and fusion weights.
            Defaults to standard LEO constellation settings.

    Example:
        >>> detector = ManeuverDetector(DetectionThresholds.for_starlink())
        >>> events = detector.detect(tles)
        >>> for event in events:
        ...     print(event.summary())
    """
    def __init__(self, thresholds: Optional[DetectionThresholds] = None) -> None:
        self.thresholds = thresholds or DetectionThresholds()

    def detect(self, tles: list[TLE]) -> list[ManeuverEvent]:
        """Detect maneuvers from a chronological TLE history.

        Args:
            tles: TLEs for a single spacecraft, in any order (will be sorted).

        Returns:
            Detected maneuver events, sorted by epoch. Closely-spaced
            detections (within 6 hours) are merged, keeping the
            highest-scoring event.
        """
        if len(tles) < 2:
            return []

        tles = sorted(tles, key=lambda t: t.epoch_dt)
        events: list[ManeuverEvent] = []

        for i in range(1, len(tles)):
            prev, curr = tles[i - 1], tles[i]

            gap_hours = (
                (curr.epoch_dt - prev.epoch_dt).total_seconds() / 3600.0
            )
            if gap_hours < self.thresholds.min_epoch_gap_hours:
                continue
            if gap_hours > self.thresholds.max_epoch_gap_days * 24.0:
                continue

            scores = self._compute_scores(prev, curr, gap_hours)
            total = self._fuse_scores(scores)

            if total >= self.thresholds.maneuver_score_threshold:
                event = self._build_event(prev, curr, gap_hours, scores, total)
                events.append(event)

        return self._merge_nearby(events)

    def _compute_scores(
        self,
        prev: TLE,
        curr: TLE,
        gap_hours: float,
    ) -> dict[str, float]:
        """Compute per-channel suspicion scores for a TLE pair.

        Each score is clamped to [0, 1], where 0 means no anomaly and 1
        means the residual exceeds the threshold.
        """
        t = self.thresholds
        gap_days = gap_hours / 24.0

        # SMA jump
        delta_sma = curr.semi_major_axis - prev.semi_major_axis
        expected_sma = self._expected_sma_drift(prev, gap_days)
        sma_residual = abs(delta_sma - expected_sma)
        score_sma = min(1.0, sma_residual / t.sma_jump_km)

        # Mean motion
        delta_mm = abs(curr.mean_motion - prev.mean_motion)
        expected_mm = abs(prev.mean_motion_dot * 2.0 * gap_days)
        mm_residual = abs(delta_mm - expected_mm)
        score_mm = min(1.0, mm_residual / t.mean_motion_jump)

        # Inclination (no J2 secular drift)
        delta_inc = abs(curr.inclination - prev.inclination)
        score_inc = min(1.0, delta_inc / t.inclination_jump_deg)

        # RAAN (after removing J2 secular drift)
        delta_raan = _angle_diff(curr.raan, prev.raan)
        expected_raan = prev.raan_rate * gap_days
        raan_residual = abs(delta_raan - expected_raan)
        if raan_residual > 180:
            raan_residual = 360 - raan_residual
        score_raan = min(1.0, raan_residual / t.raan_residual_deg)

        # Eccentricity
        delta_ecc = abs(curr.eccentricity - prev.eccentricity)
        score_ecc = min(1.0, delta_ecc / t.eccentricity_jump)

        # B* anomaly
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

    def _fuse_scores(self, scores: dict[str, float]) -> float:
        """Compute weighted sum of per-channel scores.
        """
        t = self.thresholds
        return (
            t.weight_sma * scores["sma"]
            + t.weight_mean_motion * scores["mean_motion"]
            + t.weight_inclination * scores["inclination"]
            + t.weight_raan * scores["raan"]
            + t.weight_eccentricity * scores["eccentricity"]
            + t.weight_bstar * scores["bstar"]
        )

    def _expected_sma_drift(self, tle: TLE, gap_days: float) -> float:
        """Estimate expected SMA change from atmospheric drag.

        Uses the TLE's mean motion derivative as a proxy for drag:
        ``Δa ≈ -2a / (3n) × Δn``.
        """
        n = tle.mean_motion * 2 * math.pi / 86400.0
        n_dot = tle.mean_motion_dot * 2.0  # TLE field is ṅ/2
        delta_n = n_dot * (2 * math.pi / 86400.0) * gap_days

        if abs(n) < 1e-10:
            return 0.0
        return -2.0 * tle.semi_major_axis / (3.0 * n) * delta_n

    def _build_event(
        self,
        prev: TLE,
        curr: TLE,
        gap_hours: float,
        scores: dict[str, float],
        total_score: float,
    ) -> ManeuverEvent:
        """Assemble a ManeuverEvent from a detection.
        """
        delta_sma = curr.semi_major_axis - prev.semi_major_axis
        delta_alt = curr.altitude - prev.altitude
        delta_inc = curr.inclination - prev.inclination
        delta_raan = _angle_diff(curr.raan, prev.raan)
        delta_ecc = curr.eccentricity - prev.eccentricity
        delta_mm = curr.mean_motion - prev.mean_motion

        expected_raan = prev.raan_rate * (gap_hours / 24.0)
        raan_residual = delta_raan - expected_raan

        estimated_dv = self._estimate_delta_v(prev, curr)
        maneuver_type = self._classify(delta_alt, delta_inc, delta_ecc, scores)

        gap_quality = 1.0 - min(
            1.0, gap_hours / (self.thresholds.max_epoch_gap_days * 24)
        )
        confidence = min(1.0, total_score * (0.5 + 0.5 * gap_quality))

        return ManeuverEvent(
            norad_id=curr.norad_id,
            spacecraft_name=curr.name or prev.name,
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

    @staticmethod
    def _estimate_delta_v(prev: TLE, curr: TLE) -> float:
        """Rough delta-v estimate from the vis-viva equation (m/s).

        Combines in-plane (from SMA change) and out-of-plane (from
        inclination change) components via RSS.
        """
        mu = 398600.4418
        a = prev.semi_major_axis
        v = math.sqrt(mu / a) * 1000  # m/s

        delta_a = abs(curr.semi_major_axis - prev.semi_major_axis)
        dv_inplane = v / 2.0 * delta_a / a if a > 0 else 0.0

        delta_i_rad = math.radians(abs(curr.inclination - prev.inclination))
        dv_outplane = v * delta_i_rad

        return math.sqrt(dv_inplane**2 + dv_outplane**2)

    @staticmethod
    def _classify(
        delta_alt: float,
        delta_inc: float,
        delta_ecc: float,
        scores: dict[str, float],
    ) -> ManeuverType:
        """Classify the maneuver type based on dominant detection channels."""
        if abs(delta_alt) > 50:
            return (
                ManeuverType.ORBIT_RAISE
                if delta_alt > 0
                else ManeuverType.DEORBIT
            )

        if scores["inclination"] > 0.5 or scores["raan"] > 0.5:
            if scores["sma"] < 0.3:
                return ManeuverType.PLANE_CHANGE

        if scores["sma"] > 0.3:
            if abs(delta_alt) < 0.06:
                return ManeuverType.ALTITUDE_MAINTENANCE
            return (
                ManeuverType.ALTITUDE_RAISE
                if delta_alt > 0
                else ManeuverType.ALTITUDE_LOWER
            )

        if scores["mean_motion"] > 0.4 and scores["sma"] < 0.3:
            return ManeuverType.PHASING

        return ManeuverType.UNKNOWN

    @staticmethod
    def _merge_nearby(
        events: list[ManeuverEvent],
        window_hours: float = 6.0,
    ) -> list[ManeuverEvent]:
        """Merge detections within a time window, keeping highest-scoring."""
        if not events:
            return events

        merged = [events[0]]
        for event in events[1:]:
            gap = (
                event.epoch_after - merged[-1].epoch_after
            ).total_seconds() / 3600.0
            if gap < window_hours:
                if event.score_total > merged[-1].score_total:
                    merged[-1] = event
            else:
                merged.append(event)

        return merged


# Batch utils
def detect_maneuvers_batch(
    tle_dict: dict[int, list[TLE]],
    thresholds: Optional[DetectionThresholds] = None,
) -> pd.DataFrame:
    """Detect maneuvers across multiple spacecraft.

    Args:
        tle_dict: Mapping of NORAD ID to TLE history.
        thresholds: Detection thresholds (defaults to standard LEO).

    Returns:
        DataFrame of all detected maneuver events, sorted by epoch.
    """
    detector = ManeuverDetector(thresholds)
    all_events: list[dict] = []

    for norad_id, tles in tle_dict.items():
        events = detector.detect(tles)
        all_events.extend(e.to_dict() for e in events)

    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    return df.sort_values("epoch_after").reset_index(drop=True)


def build_element_history(tles: list[TLE]) -> pd.DataFrame:
    """Convert a TLE list to a time-series DataFrame of orbital elements.

    Useful for plotting element evolution and visually confirming
    maneuver detections.

    Args:
        tles: TLEs for a single spacecraft (any order).

    Returns:
        DataFrame with one row per TLE, sorted by epoch.
    """
    records = [tle.to_dict() for tle in tles]
    df = pd.DataFrame(records)
    return df.sort_values("epoch").reset_index(drop=True)


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference ``a - b``, wrapped to [-180, 180].
    """
    d = (a - b) % 360
    return d - 360 if d > 180 else d
