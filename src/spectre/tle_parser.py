"""TLE parsing and orbital element extraction.

Parses standard NORAD Two-Line Element sets and computes derived orbital
quantities needed for maneuver detection: semi-major axis, altitude,
orbital period, inclination, RAAN secular drift rate, and more.

References:
    - Kelso, T.S. "CelesTrak TLE Format Documentation"
      https://celestrak.org/columns/v04n03/
    - Vallado, D. (2013). Fundamentals of Astrodynamics and Applications.

Author:
    Kyle Hughes (@huqhesy) — kyle.evan.hughes@gmail.com
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Physical constants (WGS84) ──

MU_EARTH = 398600.4418
"""Earth gravitational parameter (km³/s²)."""

R_EARTH = 6378.137
"""Earth equatorial radius (km)."""

J2 = 1.08262668e-3
"""Earth J2 zonal harmonic coefficient."""

SOLAR_DAY = 86400.0
"""Seconds in a solar day."""

TWO_PI = 2.0 * math.pi
"""2π constant."""


@dataclass(slots=True)
class TLE:
    """A parsed Two-Line Element set with derived orbital quantities.

    Attributes:
        name: Spacecraft name from line 0 (if present).
        norad_id: NORAD catalog number.
        intl_designator: International designator (launch year/number/piece).
        classification: Security classification (U/C/S).
        epoch_year: Full 4-digit epoch year.
        epoch_day: Fractional day of year at epoch.
        epoch_dt: Epoch as a Python datetime.
        mean_motion_dot: First derivative of mean motion / 2 (rev/day²).
        mean_motion_ddot: Second derivative of mean motion / 6 (rev/day³).
        bstar: B* drag term (1/Earth radii).
        inclination: Orbital inclination (degrees).
        raan: Right ascension of ascending node (degrees).
        eccentricity: Orbital eccentricity (dimensionless).
        arg_perigee: Argument of perigee (degrees).
        mean_anomaly: Mean anomaly (degrees).
        mean_motion: Mean motion (revolutions per day).
        rev_number: Revolution number at epoch.
        semi_major_axis: Derived semi-major axis (km).
        altitude: Derived altitude above Earth's surface (km).
        period: Derived orbital period (seconds).
        raan_rate: J2 secular RAAN drift rate (deg/day).
    """

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
    mean_motion_dot: float
    mean_motion_ddot: float
    bstar: float

    # Line 2 fields
    inclination: float
    raan: float
    eccentricity: float
    arg_perigee: float
    mean_anomaly: float
    mean_motion: float
    rev_number: int

    # Derived (computed in __post_init__)
    semi_major_axis: float = field(init=False)
    altitude: float = field(init=False)
    period: float = field(init=False)
    raan_rate: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived orbital quantities from TLE fields."""
        n_rad_s = self.mean_motion * TWO_PI / SOLAR_DAY
        self.semi_major_axis = (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0)
        self.altitude = self.semi_major_axis - R_EARTH
        self.period = SOLAR_DAY / self.mean_motion

        # J2 secular RAAN regression rate
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
    def parse(
        line1: str,
        line2: str,
        name: Optional[str] = None,
    ) -> TLE:
        """Parse a TLE from line 1 and line 2 strings.

        Args:
            line1: TLE line 1 (69 characters, starts with '1').
            line2: TLE line 2 (69 characters, starts with '2').
            name: Optional spacecraft name (from line 0).

        Returns:
            Parsed TLE object with derived orbital quantities.

        Raises:
            ValueError: If line format is invalid or NORAD IDs don't match.
        """
        l1 = line1.ljust(69)
        l2 = line2.ljust(69)

        if l1[0] != "1":
            raise ValueError(f"Line 1 must start with '1', got '{l1[0]}'")
        if l2[0] != "2":
            raise ValueError(f"Line 2 must start with '2', got '{l2[0]}'")

        _verify_checksum(l1, 1)
        _verify_checksum(l2, 2)

        # ── Line 1 ──
        norad_id = int(l1[2:7].strip())
        classification = l1[7]
        intl_designator = l1[9:17].strip()

        epoch_year_2d = int(l1[18:20].strip())
        epoch_year = (
            1900 + epoch_year_2d if epoch_year_2d >= 57 else 2000 + epoch_year_2d
        )
        epoch_day = float(l1[20:32].strip())
        mean_motion_dot = float(l1[33:43].strip())
        mean_motion_ddot = _parse_implied_decimal(l1[44:52])
        bstar = _parse_implied_decimal(l1[53:61])

        # ── Line 2 ──
        norad_id_2 = int(l2[2:7].strip())
        if norad_id != norad_id_2:
            raise ValueError(
                f"NORAD ID mismatch: {norad_id} vs {norad_id_2}"
            )

        inclination = float(l2[8:16].strip())
        raan_val = float(l2[17:25].strip())
        eccentricity = float(f"0.{l2[26:33].strip()}")
        arg_perigee = float(l2[34:42].strip())
        mean_anomaly = float(l2[42:51].strip())
        mean_motion_val = float(l2[52:63].strip())
        rev_number = int(l2[63:68].strip() or "0")

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
            raan=raan_val,
            eccentricity=eccentricity,
            arg_perigee=arg_perigee,
            mean_anomaly=mean_anomaly,
            mean_motion=mean_motion_val,
            rev_number=rev_number,
        )

    @staticmethod
    def parse_batch(text: str) -> list[TLE]:
        """Parse a multi-TLE string containing 2-line or 3-line format TLEs.

        Automatically detects whether each TLE has a name line (line 0)
        or is a bare 2-line element set.

        Args:
            text: String containing one or more TLEs separated by newlines.

        Returns:
            List of parsed TLE objects, in the order they appear.
        """
        lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
        tles: list[TLE] = []
        i = 0

        while i < len(lines):
            if (
                lines[i].startswith("1")
                and i + 1 < len(lines)
                and lines[i + 1].startswith("2")
            ):
                tles.append(TLE.parse(lines[i], lines[i + 1]))
                i += 2
            elif (
                i + 2 < len(lines)
                and lines[i + 1].startswith("1")
                and lines[i + 2].startswith("2")
            ):
                tles.append(
                    TLE.parse(lines[i + 1], lines[i + 2], name=lines[i])
                )
                i += 3
            else:
                i += 1

        return tles

    def to_dict(self) -> dict:
        """Convert to a flat dictionary suitable for DataFrame construction.

        Returns:
            Dictionary with all TLE fields and derived quantities.
        """
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


# ── Private helpers ──


def _parse_implied_decimal(s: str) -> float:
    """Parse TLE implied-decimal notation into a float.

    The TLE format encodes some fields as ``NNNNN±N`` where the mantissa
    has an implied leading ``0.`` and the final ``±N`` is a base-10
    exponent. For example, ``16538-4`` becomes ``0.16538e-4``.

    Args:
        s: Raw field string from a TLE line.

    Returns:
        Parsed floating-point value.
    """
    s = s.strip()
    if not s or s in ("00000-0", "00000+0"):
        return 0.0

    for i in range(len(s) - 1, 0, -1):
        if s[i] in "+-":
            mantissa = s[:i]
            exponent = s[i:]
            sign = "-" if mantissa.lstrip().startswith("-") else ""
            digits = mantissa.lstrip("+-").lstrip()
            return float(f"{sign}0.{digits}e{exponent}")

    sign = "-" if s.startswith("-") else ""
    digits = s.lstrip("+-").lstrip()
    return float(f"{sign}0.{digits}")


def _verify_checksum(line: str, line_num: int) -> None:
    """Verify a TLE line's modulo-10 checksum.

    Logs a warning on mismatch rather than raising — many real-world TLE
    sources have minor formatting differences, and we prefer to parse
    rather than reject.

    Args:
        line: Full 69-character TLE line.
        line_num: Line number (1 or 2) for the warning message.
    """
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
        logger.warning(
            "Checksum mismatch on line %d: expected %d, computed %d",
            line_num,
            expected,
            computed,
        )


def _epoch_to_datetime(year: int, day_of_year: float) -> datetime:
    """Convert a TLE epoch (year + fractional day-of-year) to a datetime.

    Args:
        year: Full 4-digit year.
        day_of_year: Fractional day of year (1.0 = midnight Jan 1).

    Returns:
        Corresponding UTC datetime.
    """
    jan1 = datetime(year, 1, 1)
    return jan1 + timedelta(days=day_of_year - 1.0)
