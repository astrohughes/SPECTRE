"""
TLE parsing and orbital element extraction.

Parses standard NORAD Two-Line Element sets and extracts derived quantities
needed for maneuver detection: semi-major axis, altitude, period,
inclination, RAAN, eccentricity, and mean motion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

# ── Constants ──

MU_EARTH = 398600.4418  # km³/s²
R_EARTH = 6378.137  # km
J2 = 1.08262668e-3
SOLAR_DAY = 86400.0
TWO_PI = 2.0 * math.pi


@dataclass(slots=True)
class TLE:
    """A parsed Two-Line Element set with derived orbital quantities."""

    # Identity
    name: Optional[str]
    norad_id: int
    intl_designator: str
    classification: str

    # Epoch
    epoch_year: int
    epoch_day: float
    epoch_dt: datetime

    # Line 1 fields
    mean_motion_dot: float  # rev/day² / 2
    mean_motion_ddot: float  # rev/day³ / 6
    bstar: float  # 1/R_earth

    # Line 2 fields
    inclination: float  # degrees
    raan: float  # degrees
    eccentricity: float  # dimensionless
    arg_perigee: float  # degrees
    mean_anomaly: float  # degrees
    mean_motion: float  # rev/day
    rev_number: int

    # Derived quantities (computed on parse)
    semi_major_axis: float = field(init=False)  # km
    altitude: float = field(init=False)  # km
    period: float = field(init=False)  # seconds
    raan_rate: float = field(init=False)  # deg/day (J2 secular)

    def __post_init__(self):
        n_rad_s = self.mean_motion * TWO_PI / SOLAR_DAY
        self.semi_major_axis = (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0)
        self.altitude = self.semi_major_axis - R_EARTH
        self.period = SOLAR_DAY / self.mean_motion

        # J2 secular RAAN rate
        i_rad = math.radians(self.inclination)
        p = self.semi_major_axis * (1.0 - self.eccentricity**2)
        if p > 0:
            self.raan_rate = (
                -1.5
                * n_rad_s
                * J2
                * (R_EARTH / p) ** 2
                * math.cos(i_rad)
                * math.degrees(1.0)
                * SOLAR_DAY
            )
        else:
            self.raan_rate = 0.0

    @staticmethod
    def parse(line1: str, line2: str, name: Optional[str] = None) -> TLE:
        """Parse a TLE from line 1 and line 2 strings."""
        l1 = line1.ljust(69)
        l2 = line2.ljust(69)

        if l1[0] != "1":
            raise ValueError(f"Line 1 must start with '1', got '{l1[0]}'")
        if l2[0] != "2":
            raise ValueError(f"Line 2 must start with '2', got '{l2[0]}'")

        # Validate checksums
        _verify_checksum(l1, 1)
        _verify_checksum(l2, 2)

        # ── Line 1 ──
        norad_id = int(l1[2:7].strip())
        classification = l1[7]
        intl_designator = l1[9:17].strip()

        epoch_year_2d = int(l1[18:20].strip())
        epoch_year = 1900 + epoch_year_2d if epoch_year_2d >= 57 else 2000 + epoch_year_2d
        epoch_day = float(l1[20:32].strip())

        mean_motion_dot = float(l1[33:43].strip())
        mean_motion_ddot = _parse_implied_decimal(l1[44:52])
        bstar = _parse_implied_decimal(l1[53:61])

        # ── Line 2 ──
        norad_id_2 = int(l2[2:7].strip())
        if norad_id != norad_id_2:
            raise ValueError(f"NORAD ID mismatch: {norad_id} vs {norad_id_2}")

        inclination = float(l2[8:16].strip())
        raan = float(l2[17:25].strip())
        eccentricity = float(f"0.{l2[26:33].strip()}")
        arg_perigee = float(l2[34:42].strip())
        mean_anomaly = float(l2[42:51].strip())
        mean_motion = float(l2[52:63].strip())
        rev_number = int(l2[63:68].strip() or "0")

        # Compute epoch datetime
        epoch_dt = _epoch_to_datetime(epoch_year, epoch_day)

        return TLE(
            name=name.strip() if name else None,
            norad_id=norad_id,
            intl_designator=intl_designator,
            classification=classification,
            epoch_year=epoch_year,
            epoch_day=epoch_day,
            epoch_dt=epoch_dt,
            mean_motion_dot=mean_motion_dot,
            mean_motion_ddot=mean_motion_ddot,
            bstar=bstar,
            inclination=inclination,
            raan=raan,
            eccentricity=eccentricity,
            arg_perigee=arg_perigee,
            mean_anomaly=mean_anomaly,
            mean_motion=mean_motion,
            rev_number=rev_number,
        )

    @staticmethod
    def parse_batch(text: str) -> list[TLE]:
        """Parse a multi-TLE string. Handles 2-line and 3-line formats."""
        lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
        tles = []
        i = 0

        while i < len(lines):
            if lines[i].startswith("1") and i + 1 < len(lines) and lines[i + 1].startswith("2"):
                tles.append(TLE.parse(lines[i], lines[i + 1]))
                i += 2
            elif (
                i + 2 < len(lines)
                and lines[i + 1].startswith("1")
                and lines[i + 2].startswith("2")
            ):
                tles.append(TLE.parse(lines[i + 1], lines[i + 2], name=lines[i]))
                i += 3
            else:
                i += 1

        return tles

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "norad_id": self.norad_id,
            "name": self.name,
            "epoch": self.epoch_dt,
            "epoch_year": self.epoch_year,
            "epoch_day": self.epoch_day,
            "sma_km": self.semi_major_axis,
            "altitude_km": self.altitude,
            "period_s": self.period,
            "inclination_deg": self.inclination,
            "raan_deg": self.raan,
            "eccentricity": self.eccentricity,
            "arg_perigee_deg": self.arg_perigee,
            "mean_anomaly_deg": self.mean_anomaly,
            "mean_motion_rev_day": self.mean_motion,
            "mean_motion_dot": self.mean_motion_dot,
            "bstar": self.bstar,
            "raan_rate_deg_day": self.raan_rate,
            "rev_number": self.rev_number,
        }


def _parse_implied_decimal(s: str) -> float:
    """Parse TLE implied-decimal format: ' NNNNN-N' → float."""
    s = s.strip()
    if not s or s in ("00000-0", "00000+0"):
        return 0.0

    # Find last +/- that isn't the leading sign
    for i in range(len(s) - 1, 0, -1):
        if s[i] in "+-":
            mantissa = s[:i]
            exponent = s[i:]
            sign = "-" if mantissa.lstrip().startswith("-") else ""
            digits = mantissa.lstrip("+-").lstrip()
            return float(f"{sign}0.{digits}e{exponent}")

    # No exponent found
    sign = "-" if s.startswith("-") else ""
    digits = s.lstrip("+-").lstrip()
    return float(f"{sign}0.{digits}")


def _verify_checksum(line: str, line_num: int):
    """
    Verify TLE line checksum.

    Logs a warning on mismatch rather than raising — many real-world TLE
    sources have minor formatting issues, and we'd rather parse than reject.
    """
    import logging
    _logger = logging.getLogger(__name__)

    if len(line) < 69 or not line[68].isdigit():
        return

    expected = int(line[68])
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1

    computed = total % 10
    if computed != expected:
        _logger.warning(
            f"Checksum mismatch on line {line_num}: expected {expected}, computed {computed}"
        )


def _epoch_to_datetime(year: int, day_of_year: float) -> datetime:
    """Convert TLE epoch (year + fractional day) to datetime."""
    jan1 = datetime(year, 1, 1)
    return jan1 + timedelta(days=day_of_year - 1.0)
