"""Space-Track.org API client for fetching TLE history.

Provides authenticated access to Space-Track's REST API for downloading
historical TLE data. Includes disk-based caching and rate limiting to
comply with Space-Track's usage policies.

Requires a free account at https://www.space-track.org/auth/createAccount

Set credentials via environment variables::

    export SPACETRACK_USER="your@email.com"
    export SPACETRACK_PASS="your_password"

Or pass them directly to the ``SpaceTrackClient`` constructor.

Author:
    Kyle Hughes (@huqhesy) — kyle.evan.hughes@gmail.com
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests

from .tle_parser import TLE

logger = logging.getLogger(__name__)

BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
QUERY_URL = f"{BASE_URL}/basicspacedata/query"

# Space-Track rate limits: 30 requests per minute, 300 per hour
RATE_LIMIT_DELAY = 2.5  # seconds between requests


class SpaceTrackClient:
    """Client for the Space-Track.org REST API."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.username = username or os.environ.get("SPACETRACK_USER", "")
        self.password = password or os.environ.get("SPACETRACK_PASS", "")
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._authenticated = False
        self._last_request_time = 0.0

    def _authenticate(self):
        """Login to Space-Track."""
        if self._authenticated:
            return

        if not self.username or not self.password:
            raise ValueError(
                "Space-Track credentials required. Set SPACETRACK_USER and "
                "SPACETRACK_PASS environment variables, or pass to constructor.\n"
                "Register free at: https://www.space-track.org/auth/createAccount"
            )

        resp = self.session.post(
            LOGIN_URL,
            data={"identity": self.username, "password": self.password},
        )
        if resp.status_code != 200 or "Login Failed" in resp.text:
            raise ConnectionError(
                f"Space-Track authentication failed (HTTP {resp.status_code})"
            )

        self._authenticated = True
        logger.info("Authenticated with Space-Track")

    def _rate_limit(self):
        """Respect Space-Track rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _query(self, endpoint: str, use_cache: bool = True) -> str:
        """Execute a Space-Track query, with optional disk caching."""
        # Check cache
        cache_key = endpoint.replace("/", "_").replace(" ", "_")[:200]
        cache_file = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 24:  # Cache valid for 24 hours
                logger.debug(f"Cache hit: {cache_file.name}")
                return cache_file.read_text()

        self._authenticate()
        self._rate_limit()

        url = f"{QUERY_URL}/{endpoint}"
        logger.info(f"Querying: {url}")

        resp = self.session.get(url)
        resp.raise_for_status()

        # Cache the response
        if use_cache:
            cache_file.write_text(resp.text)

        return resp.text

    def get_tle_history(
        self,
        norad_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 5000,
    ) -> list[TLE]:
        """
        Fetch TLE history for a single satellite.

        Args:
            norad_id: NORAD catalog number
            start_date: Start of date range (default: 90 days ago)
            end_date: End of date range (default: now)
            limit: Maximum number of TLEs to fetch

        Returns:
            List of TLE objects sorted by epoch.
        """
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        date_range = (
            f"{start_date.strftime('%Y-%m-%d')}--{end_date.strftime('%Y-%m-%d')}"
        )

        endpoint = (
            f"class/gp_history/NORAD_CAT_ID/{norad_id}/"
            f"EPOCH/{date_range}/"
            f"orderby/EPOCH asc/limit/{limit}/format/tle"
        )

        raw = self._query(endpoint)
        if not raw.strip():
            logger.warning(f"No TLEs found for NORAD {norad_id}")
            return []

        return TLE.parse_batch(raw)

    def get_constellation_tles(
        self,
        norad_ids: list[int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit_per_sat: int = 2000,
    ) -> dict[int, list[TLE]]:
        """
        Fetch TLE history for multiple satellites.

        Args:
            norad_ids: List of NORAD catalog numbers
            start_date: Start of date range
            end_date: End of date range
            limit_per_sat: Max TLEs per satellite

        Returns:
            Dict mapping NORAD ID → list of TLEs.
        """
        from tqdm import tqdm

        results = {}
        for norad_id in tqdm(norad_ids, desc="Fetching TLEs"):
            try:
                tles = self.get_tle_history(
                    norad_id, start_date, end_date, limit_per_sat
                )
                if tles:
                    results[norad_id] = tles
            except Exception as e:
                logger.warning(f"Failed to fetch NORAD {norad_id}: {e}")

        return results

    def search_by_name(self, name_pattern: str, limit: int = 100) -> list[dict]:
        """
        Search for satellites by name pattern.

        Returns basic catalog info (NORAD ID, name, launch date, etc.)
        """
        endpoint = (
            f"class/gp/OBJECT_NAME/~~{name_pattern}/"
            f"orderby/NORAD_CAT_ID asc/limit/{limit}/format/json"
        )

        raw = self._query(endpoint, use_cache=False)
        return json.loads(raw)

    def get_latest_tle(self, norad_id: int) -> Optional[TLE]:
        """Fetch the most recent TLE for a satellite."""
        endpoint = (
            f"class/gp/NORAD_CAT_ID/{norad_id}/"
            f"orderby/EPOCH desc/limit/1/format/tle"
        )

        raw = self._query(endpoint, use_cache=False)
        tles = TLE.parse_batch(raw) if raw.strip() else []
        return tles[0] if tles else None


def load_tle_file(filepath: str | Path) -> list[TLE]:
    """Load TLEs from a local file (2-line or 3-line format)."""
    text = Path(filepath).read_text()
    return TLE.parse_batch(text)
