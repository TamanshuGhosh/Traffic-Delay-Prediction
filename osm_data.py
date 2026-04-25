"""
osm_data.py

Builds a road-segment dataset from OpenStreetMap road graphs.

What it does:
- Downloads or loads an OSMnx graph for a place
- Extracts edge-level features
- Creates a "smart hack" synthetic historical dataset by sampling
  multiple time contexts per road segment
- Optionally enriches delay estimates with HERE traffic if available

Outputs:
- A CSV dataset suitable for training a delay prediction model
- A cached GraphML file for the place

Dependencies:
    osmnx, networkx, pandas, numpy, joblib, requests (optional for HERE)
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import osmnx as ox

try:
    from here_api import get_live_traffic_factor
except Exception:
    get_live_traffic_factor = None


DATA_DIR = Path("data")
GRAPH_DIR = DATA_DIR / "graphs"
DATA_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
ROAD_TYPE_ORDER = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "other",
]


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def _clean_list_like(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def parse_lanes(value: Any, default: float = 1.0) -> float:
    value = _clean_list_like(value)
    if value is None:
        return float(default)

    if isinstance(value, (int, float)) and not pd.isna(value):
        return max(float(value), 1.0)

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "unknown"}:
        return float(default)

    # examples: "2", "2;1", "2|1", "1,2", "3 lanes"
    nums = []
    for token in re_split_numbers(text):
        try:
            nums.append(float(token))
        except Exception:
            pass

    if nums:
        return max(float(np.mean(nums)), 1.0)

    return float(default)


def re_split_numbers(text: str) -> list[str]:
    import re
    return re.findall(r"\d+(?:\.\d+)?", text)


def parse_maxspeed_kph(value: Any, highway: str, lanes: float) -> float:
    value = _clean_list_like(value)
    if value is not None:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)
        text = str(value).lower()
        nums = re_split_numbers(text)
        if nums:
            try:
                return float(nums[0])
            except Exception:
                pass

    # Fallback heuristic
    highway = (highway or "other").lower()
    if "motorway" in highway:
        base = 80.0
    elif "trunk" in highway:
        base = 65.0
    elif "primary" in highway:
        base = 50.0
    elif "secondary" in highway:
        base = 40.0
    elif "tertiary" in highway:
        base = 35.0
    elif "residential" in highway:
        base = 25.0
    elif "service" in highway:
        base = 20.0
    else:
        base = 30.0

    # small lane bump
    return base + max(0.0, lanes - 1.0) * 3.0


def normalize_road_type(highway: Any) -> str:
    highway = _clean_list_like(highway)
    if highway is None:
        return "other"
    if isinstance(highway, (list, tuple)) and highway:
        highway = highway[0]
    text = str(highway).lower()

    for label in ROAD_TYPE_ORDER[:-1]:
        if label in text:
            return label
    return "other"


def is_peak_hour(hour: int) -> int:
    return int((7 <= hour <= 10) or (17 <= hour <= 20))


def congestion_multiplier(hour: int, day_of_week: int, road_type: str) -> float:
    """
    A realistic-but-synthetic congestion shape:
    - peak hours are slower
    - weekends are mildly smoother
    - arterial roads slow more in peaks
    """
    peak = 1.0
    if 7 <= hour <= 10:
        peak = 1.45
    elif 17 <= hour <= 20:
        peak = 1.60
    elif 21 <= hour <= 23:
        peak = 1.12
    elif 0 <= hour <= 5:
        peak = 0.95
    else:
        peak = 1.08

    weekend = 0.92 if day_of_week in (5, 6) else 1.0

    road_bias = {
        "motorway": 0.95,
        "trunk": 1.00,
        "primary": 1.08,
        "secondary": 1.10,
        "tertiary": 1.05,
        "residential": 1.03,
        "service": 0.98,
        "other": 1.00,
    }.get(road_type, 1.0)

    return peak * weekend * road_bias


def road_priority_bias(road_type: str, lanes: float) -> float:
    """
    Mildly lowers delay on roads that usually move better.
    """
    type_bias = {
        "motorway": 0.85,
        "trunk": 0.92,
        "primary": 1.00,
        "secondary": 1.03,
        "tertiary": 1.04,
        "residential": 1.08,
        "service": 1.10,
        "other": 1.00,
    }.get(road_type, 1.0)

    lane_bias = max(0.88, 1.0 - 0.025 * max(lanes - 1.0, 0.0))
    return type_bias * lane_bias


def estimate_edge_base_time_minutes(length_m: float, speed_kph: float) -> float:
    speed_mps = max(speed_kph, 1.0) * 1000.0 / 3600.0
    return float(length_m) / speed_mps / 60.0


def graph_cache_path(place: str, network_type: str = "drive") -> Path:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in place).strip("_")
    return GRAPH_DIR / f"{slug}_{network_type}.graphml"


def load_or_download_graph(place: str, network_type: str = "drive"):
    cache_path = graph_cache_path(place, network_type)
    if cache_path.exists():
        return ox.load_graphml(cache_path)
    G = ox.graph_from_place(place, network_type=network_type, simplify=True)
    ox.save_graphml(G, cache_path)
    return G


def edge_static_features(
    u: Any,
    v: Any,
    key: Any,
    attr: Dict[str, Any],
) -> Dict[str, Any]:
    highway = normalize_road_type(attr.get("highway", "other"))
    lanes = parse_lanes(attr.get("lanes", 1), default=1.0)
    length_m = float(attr.get("length", 100.0) or 100.0)
    maxspeed_kph = parse_maxspeed_kph(attr.get("maxspeed"), highway, lanes)

    base_time_min = estimate_edge_base_time_minutes(length_m, maxspeed_kph)

    return {
        "u": u,
        "v": v,
        "key": key,
        "osmid": str(attr.get("osmid", "")),
        "highway": highway,
        "lanes": lanes,
        "length_m": length_m,
        "maxspeed_kph": maxspeed_kph,
        "base_time_min": base_time_min,
        "oneway": int(bool(attr.get("oneway", False))),
        "bridge": int(bool(attr.get("bridge", False))),
        "tunnel": int(bool(attr.get("tunnel", False))),
        "junction": str(_clean_list_like(attr.get("junction", "none")) or "none"),
        "access": str(_clean_list_like(attr.get("access", "unknown")) or "unknown"),
        "service": str(_clean_list_like(attr.get("service", "none")) or "none"),
    }


def synthetic_delay_minutes(
    static_row: Dict[str, Any],
    hour: int,
    day_of_week: int,
    live_factor: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()

    base = float(static_row["base_time_min"])
    highway = static_row["highway"]
    lanes = float(static_row["lanes"])

    congestion = congestion_multiplier(hour, day_of_week, highway)
    priority = road_priority_bias(highway, lanes)

    if live_factor is None:
        live_factor = 1.0

    # Small random realism so the model learns a noisy relationship
    noise = float(rng.normal(loc=1.0, scale=0.06))
    noise = float(np.clip(noise, 0.85, 1.18))

    actual_time = base * congestion * priority * live_factor * noise
    delay = max(0.0, actual_time - base)
    return float(delay)


def build_dataset(
    place: str,
    network_type: str = "drive",
    samples_per_edge: int = 8,
    use_here_live_data: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Any]:
    """
    Builds a synthetic historical dataset around the road graph.

    Strategy:
    - Extract each road edge
    - Sample several time contexts per edge
    - Simulate observed delay using time-of-day patterns
    - Optionally enrich with HERE live traffic factor if configured

    Returns:
        (dataframe, graph)
    """
    rng = np.random.default_rng(random_state)
    py_rng = random.Random(random_state)

    G = load_or_download_graph(place, network_type=network_type)

    rows = []
    for u, v, key, attr in G.edges(keys=True, data=True):
        static_row = edge_static_features(u, v, key, attr)

        for _ in range(samples_per_edge):
            hour = int(py_rng.randint(0, 23))
            day_of_week = int(py_rng.randint(0, 6))
            minute_bucket = int(py_rng.choice([0, 15, 30, 45]))

            live_factor = None
            if use_here_live_data and callable(get_live_traffic_factor):
                try:
                    # Optional hook. If your HERE wrapper returns a factor, use it.
                    # Otherwise it will stay as None and the synthetic fallback applies.
                    live_factor = get_live_traffic_factor(
                        place=place,
                        highway=static_row["highway"],
                        length_m=static_row["length_m"],
                        hour=hour,
                        day_of_week=day_of_week,
                    )
                except Exception:
                    live_factor = None

            delay_min = synthetic_delay_minutes(
                static_row=static_row,
                hour=hour,
                day_of_week=day_of_week,
                live_factor=live_factor,
                rng=rng,
            )

            row = dict(static_row)
            row.update(
                {
                    "hour": hour,
                    "minute_bucket": minute_bucket,
                    "day_of_week": day_of_week,
                    "is_peak": is_peak_hour(hour),
                    "live_factor": 1.0 if live_factor is None else float(live_factor),
                    "delay_min": delay_min,
                    "observed_time_min": float(static_row["base_time_min"] + delay_min),
                    "congestion_ratio": float(
                        (static_row["base_time_min"] + delay_min) / max(static_row["base_time_min"], 1e-6)
                    ),
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    # Keep dataset tidy
    categorical_cols = ["highway", "junction", "access", "service"]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    return df, G


def save_dataset(
    place: str,
    out_csv: str = "data/road_delay_dataset.csv",
    network_type: str = "drive",
    samples_per_edge: int = 8,
    use_here_live_data: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Any, Path]:
    df, G = build_dataset(
        place=place,
        network_type=network_type,
        samples_per_edge=samples_per_edge,
        use_here_live_data=use_here_live_data,
        random_state=random_state,
    )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df, G, out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a road delay dataset from OSM.")
    parser.add_argument("--place", type=str, default="Kolkata, India")
    parser.add_argument("--network_type", type=str, default="drive")
    parser.add_argument("--samples_per_edge", type=int, default=8)
    parser.add_argument("--out_csv", type=str, default="data/road_delay_dataset.csv")
    parser.add_argument("--no_here", action="store_true", help="Disable HERE live traffic hook")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    df, G, out_path = save_dataset(
        place=args.place,
        out_csv=args.out_csv,
        network_type=args.network_type,
        samples_per_edge=args.samples_per_edge,
        use_here_live_data=not args.no_here,
        random_state=args.random_state,
    )

    print(f"Saved {len(df):,} rows to {out_path}")
    print(f"Graph nodes: {len(G.nodes):,} | edges: {len(G.edges):,}")
