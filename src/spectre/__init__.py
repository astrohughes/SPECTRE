"""SPECTRE — Spacecraft Propulsive Event Classification & Tracking from Repeated Elements.

Detect orbital maneuvers from publicly available Two-Line Element (TLE) data.
Built for OSINT analysis of spacecraft constellation operations.

Author:
    Kyle Hughes (@astrohughes) — kyle.evan.hughes@gmail.com

Modules:
    tle_parser:     Parse and extract orbital elements from TLE sets.
    detector:       Core maneuver detection engine with multi-channel fusion.
    spacetrack:     Space-Track.org API client with caching and rate limiting.
    constellations: Presets and metadata for known spacecraft constellations.
    viz:            Visualization tools for element history and detections.
    cli:            Command-line interface.

Example:
    >>> from spectre.tle_parser import TLE
    >>> from spectre.detector import ManeuverDetector
    >>>
    >>> tles = TLE.parse_batch(open("catalog.tle").read())
    >>> detector = ManeuverDetector()
    >>> events = detector.detect(tles)
    >>> for event in events:
    ...     print(event.summary())
"""

__version__ = "0.1.0"
__author__ = "Kyle Hughes"
__email__ = "kyle.evan.hughes@gmail.com"
