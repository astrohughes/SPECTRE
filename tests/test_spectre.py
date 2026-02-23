#!/usr/bin/env python3
"""
Unit test driver for SPECTRE: Spacecraft Propulsive Event Classification & Tracking from Repeated Elements

TODO: Add scenario tests; for now this just ensures the test framework is set up correctly.
"""
import pytest
import math
from datetime import datetime, timedelta

from spectre.tle_parser import TLE
from spectre.detector import (
    ManeuverDetector,
    DetectionThresholds,
    ManeuverType,
    build_element_history,
    detect_maneuvers_batch,
)


# TLE PARSER TESTS
ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003"
ISS_LINE2 = "2 25544  51.6400 208.5000 0007417  68.0000 292.1000 15.49560000400000"


class TestTLEParser:
    def test_parse_iss(self):
        tle = TLE.parse(ISS_LINE1, ISS_LINE2)
        assert tle.norad_id == 25544
        assert tle.classification == "U"
        assert tle.epoch_year == 2024
        assert abs(tle.epoch_day - 1.5) < 1e-8
        assert abs(tle.inclination - 51.64) < 0.01
        assert abs(tle.raan - 208.5) < 0.01
        assert abs(tle.eccentricity - 0.0007417) < 1e-8
        assert abs(tle.mean_motion - 15.4956) < 0.001

    def test_altitude_iss(self):
        tle = TLE.parse(ISS_LINE1, ISS_LINE2)
        assert 400 < tle.altitude < 430

    def test_period_iss(self):
        tle = TLE.parse(ISS_LINE1, ISS_LINE2)
        # ISS period ~92.6 minutes
        assert 5500 < tle.period < 5600

    def test_parse_3line(self):
        tle = TLE.parse(ISS_LINE1, ISS_LINE2, name="ISS (ZARYA)")
        assert tle.name == "ISS (ZARYA)"

    def test_parse_batch(self):
        text = f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}\n"
        tles = TLE.parse_batch(text)
        assert len(tles) == 1
        assert tles[0].name == "ISS (ZARYA)"

    def test_parse_batch_multiple(self):
        text = (
            f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}\n"
            f"HUBBLE\n"
            f"1 20580U 90037B   24001.50000000  .00000764  00000-0  34340-4 0  9998\n"
            f"2 20580  28.4700 100.2000 0002500 300.0000  60.0000 15.09000000400000\n"
        )
        tles = TLE.parse_batch(text)
        assert len(tles) == 2

    def test_epoch_datetime(self):
        tle = TLE.parse(ISS_LINE1, ISS_LINE2)
        # 2024, day 1.5 = Jan 1 at noon
        assert tle.epoch_dt.year == 2024
        assert tle.epoch_dt.month == 1
        assert tle.epoch_dt.day == 1
        assert tle.epoch_dt.hour == 12

    def test_to_dict(self):
        tle = TLE.parse(ISS_LINE1, ISS_LINE2)
        d = tle.to_dict()
        assert d["norad_id"] == 25544
        assert "sma_km" in d
        assert "altitude_km" in d

    def test_invalid_line1_start(self):
        with pytest.raises(ValueError, match="Line 1 must start"):
            TLE.parse("2 25544..." + " " * 60, ISS_LINE2)

    def test_norad_id_mismatch(self):
        bad_line2 = "2 99999" + ISS_LINE2[7:]
        with pytest.raises(ValueError, match="NORAD ID mismatch"):
            TLE.parse(ISS_LINE1, bad_line2)


# ═══════════════════════════════════════════════════════════════
# MANEUVER DETECTION TESTS
# ═══════════════════════════════════════════════════════════════
def _make_tle(
    norad_id: int = 55001,
    epoch_dt: datetime = datetime(2024, 6, 1),
    mean_motion: float = 15.058,
    inclination: float = 53.0,
    raan: float = 0.0,
    eccentricity: float = 0.00015,
    bstar: float = 6.2e-5,
    mean_motion_dot: float = 0.0000012,
    name: str = "TEST-SAT",
) -> TLE:
    """Create a synthetic TLE for testing."""
    epoch_day = (epoch_dt - datetime(epoch_dt.year, 1, 1)).total_seconds() / 86400.0 + 1.0
    return TLE(
        name=name,
        norad_id=norad_id,
        intl_designator="24001A",
        classification="U",
        epoch_year=epoch_dt.year,
        epoch_day=epoch_day,
        epoch_dt=epoch_dt,
        mean_motion_dot=mean_motion_dot,
        mean_motion_ddot=0.0,
        bstar=bstar,
        inclination=inclination,
        raan=raan,
        eccentricity=eccentricity,
        arg_perigee=90.0,
        mean_anomaly=0.0,
        mean_motion=mean_motion,
        rev_number=1000,
    )


class TestManeuverDetector:
    def test_no_maneuver_in_stable_orbit(self):
        """Steady-state TLEs with small natural variation → no detection."""
        base_dt = datetime(2024, 6, 1)
        tles = []
        for i in range(20):
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=12 * i),
                mean_motion=15.058 + i * 0.00001,  # Tiny natural drag
                raan=0.0 + i * (-5.0 * 12 / 24),  # Expected J2 RAAN drift
            ))

        detector = ManeuverDetector()
        events = detector.detect(tles)
        assert len(events) == 0

    def test_detect_altitude_raise(self):
        """Sudden SMA/mean-motion jump → altitude raise detection."""
        base_dt = datetime(2024, 6, 1)
        tles = []

        # 5 TLEs at ~550 km
        for i in range(5):
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=12 * i),
                mean_motion=15.058,
            ))

        # Jump: mean motion drops (altitude increases)
        for i in range(5, 10):
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=12 * i),
                mean_motion=15.050,  # Lower n → higher altitude
            ))

        detector = ManeuverDetector()
        events = detector.detect(tles)

        assert len(events) >= 1
        # The maneuver should be detected at the transition point
        event = events[0]
        assert event.delta_altitude_km > 0
        assert event.maneuver_type in (
            ManeuverType.ALTITUDE_RAISE,
            ManeuverType.ALTITUDE_MAINTENANCE,
            ManeuverType.ORBIT_RAISE,
            ManeuverType.UNKNOWN,
        )

    def test_detect_altitude_lower(self):
        """SMA decrease → altitude lowering."""
        base_dt = datetime(2024, 6, 1)
        tles = []

        for i in range(5):
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=12 * i),
                mean_motion=15.050,
            ))

        # Jump: mean motion increases (altitude drops)
        for i in range(5, 10):
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=12 * i),
                mean_motion=15.060,
            ))

        detector = ManeuverDetector()
        events = detector.detect(tles)

        assert len(events) >= 1
        assert events[0].delta_altitude_km < 0

    def test_detect_plane_change(self):
        """Inclination jump → plane change detection."""
        base_dt = datetime(2024, 6, 1)
        tles = []

        for i in range(5):
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=12 * i),
                inclination=53.0,
                mean_motion=15.058,
            ))

        for i in range(5, 10):
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=12 * i),
                inclination=53.05,  # 0.05° jump
                mean_motion=15.058,
            ))

        detector = ManeuverDetector()
        events = detector.detect(tles)

        assert len(events) >= 1
        assert abs(events[0].delta_inclination_deg) > 0.01

    def test_merge_nearby_detections(self):
        """Multiple detections within 6 hours → merged into one."""
        base_dt = datetime(2024, 6, 1)
        tles = []

        # Create closely-spaced TLEs around a maneuver
        for i in range(10):
            mm = 15.058 if i < 5 else 15.048  # Jump at index 5
            tles.append(_make_tle(
                epoch_dt=base_dt + timedelta(hours=3 * i),  # 3-hour spacing
                mean_motion=mm,
            ))

        detector = ManeuverDetector()
        events = detector.detect(tles)

        # Should merge into at most 1-2 events
        assert len(events) <= 2

    def test_respects_epoch_gap_limits(self):
        """TLEs with gaps > max_epoch_gap_days are skipped."""
        base_dt = datetime(2024, 6, 1)
        tles = [
            _make_tle(epoch_dt=base_dt, mean_motion=15.058),
            _make_tle(epoch_dt=base_dt + timedelta(days=30), mean_motion=15.048),
        ]

        detector = ManeuverDetector(DetectionThresholds(max_epoch_gap_days=5.0))
        events = detector.detect(tles)
        assert len(events) == 0

    def test_delta_v_estimate_positive(self):
        """Estimated delta-v should be positive for any maneuver."""
        base_dt = datetime(2024, 6, 1)
        tles = [
            _make_tle(epoch_dt=base_dt, mean_motion=15.058),
            _make_tle(epoch_dt=base_dt + timedelta(hours=12), mean_motion=15.048),
        ]

        detector = ManeuverDetector()
        events = detector.detect(tles)

        if events:
            assert events[0].estimated_dv_ms > 0

    def test_custom_thresholds(self):
        """Custom thresholds change detection sensitivity."""
        base_dt = datetime(2024, 6, 1)
        tles = [
            _make_tle(epoch_dt=base_dt, mean_motion=15.058),
            _make_tle(epoch_dt=base_dt + timedelta(hours=12), mean_motion=15.060),
        ]

        # Very sensitive thresholds
        sensitive = DetectionThresholds(
            sma_jump_km=0.01,
            mean_motion_jump=0.001,
            maneuver_score_threshold=0.1,
        )
        events_sensitive = ManeuverDetector(sensitive).detect(tles)

        # Very insensitive thresholds
        insensitive = DetectionThresholds(
            sma_jump_km=10.0,
            mean_motion_jump=1.0,
            maneuver_score_threshold=0.9,
        )
        events_insensitive = ManeuverDetector(insensitive).detect(tles)

        assert len(events_sensitive) >= len(events_insensitive)


class TestBatchDetection:
    def test_batch_returns_dataframe(self):
        base_dt = datetime(2024, 6, 1)
        tles_a = [
            _make_tle(norad_id=55001, epoch_dt=base_dt, mean_motion=15.058),
            _make_tle(norad_id=55001, epoch_dt=base_dt + timedelta(hours=12), mean_motion=15.048),
        ]
        tles_b = [
            _make_tle(norad_id=55002, epoch_dt=base_dt, mean_motion=15.058),
            _make_tle(norad_id=55002, epoch_dt=base_dt + timedelta(hours=12), mean_motion=15.058),
        ]

        df = detect_maneuvers_batch({55001: tles_a, 55002: tles_b})
        # Only sat 55001 should have a detection
        assert len(df) >= 1
        assert 55001 in df["norad_id"].values

    def test_element_history(self):
        base_dt = datetime(2024, 6, 1)
        tles = [
            _make_tle(epoch_dt=base_dt + timedelta(hours=i * 12))
            for i in range(10)
        ]

        df = build_element_history(tles)
        assert len(df) == 10
        assert "sma_km" in df.columns
        assert "altitude_km" in df.columns
        assert "inclination_deg" in df.columns


class TestPresets:
    def test_starlink_thresholds(self):
        t = DetectionThresholds.for_starlink()
        assert t.sma_jump_km > 0
        assert t.maneuver_score_threshold > 0

    def test_oneweb_thresholds(self):
        t = DetectionThresholds.for_oneweb()
        assert t.sma_jump_km > 0

    def test_iridium_thresholds(self):
        t = DetectionThresholds.for_iridium()
        assert t.sma_jump_km > 0
