"""
here_api.py

Optional HERE traffic enrichment.

This file is intentionally defensive:
- If you have no HERE credentials, it returns None
- If the API call fails, it returns None
- The rest of the project still works with synthetic congestion data

You can use it later to plug in real traffic behavior without changing
the training or app code.

Environment variables:
- HERE_API_KEY: your HERE API key
- HERE_BASE_URL: optional override
"""

from __future__ import annotations

import os
from typing import Optional

import requests


def get_here_api_key() -> Optional[str]:
    key = os.getenv("HERE_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def get_live_traffic_factor(
    place: str,
    highway: str,
    length_m: float,
    hour: int,
    day_of_week: int,
) -> Optional[float]:
    """
    Best-effort hook for a future live traffic integration.

    Current behavior:
    - Returns None unless HERE_API_KEY exists
    - If you later connect a real HERE traffic endpoint, replace the
      request logic here

    The rest of the code is built to accept a multiplicative factor:
    - 1.0 = normal
    - >1.0 = slower / more congested
    - <1.0 = smoother
    """
    api_key = get_here_api_key()
    if not api_key:
        return None

    # Placeholder implementation:
    # You can replace this with a real HERE Traffic or Routing API query.
    # Keeping it conservative prevents the whole project from breaking
    # if you do not have a paid/live endpoint.
    try:
        # Small heuristic so the model can be fed with a real-time-like factor
        peak = 1.15 if (7 <= hour <= 10 or 17 <= hour <= 20) else 1.0
        road_bias = 1.05 if highway in {"primary", "secondary", "tertiary"} else 1.0
        weekend = 0.95 if day_of_week in {5, 6} else 1.0

        # Optionally perform a harmless sanity request if you want to validate the key.
        # We do not depend on the response.
        _ = api_key  # keep variable used
        factor = peak * road_bias * weekend
        return float(factor)
    except Exception:
        return None
