"""
run_app.py

Streamlit app for:
- loading an OSM road graph for a chosen place
- predicting delay on each road segment using the trained XGBoost model
- folding the predicted delay into graph weights
- finding the fastest route using predicted travel time
- showing route-level and edge-level predictions in the UI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st

from osm_data import edge_static_features, load_or_download_graph

try:
    from here_api import get_live_traffic_factor
except Exception:
    get_live_traffic_factor = None

st.set_page_config(page_title="Smart Traffic Delay Prediction", layout="wide")
st.title("🚦 Smart Traffic Delay Prediction & Routing")

MODEL_PATH = Path("model/delay_model.pkl")

CITIES = ["Kolkata", "Bangalore", "Hyderabad", "Delhi", "Mumbai"]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Train the model first using train_model.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_graph(place: str, network_type: str):
    return load_or_download_graph(place, network_type=network_type)


def safe_geocode(place_text: str):
    try:
        return ox.geocode(place_text)
    except Exception:
        return None


def is_peak_hour(hour: int) -> int:
    return int((7 <= hour <= 10) or (17 <= hour <= 20))


def build_feature_frame(
    static_row: Dict[str, Any],
    hour: int,
    minute_bucket: int,
    day_of_week: int,
    city: str,
    live_factor: float,
) -> pd.DataFrame:
    row = {
        "length_m": static_row["length_m"],
        "lanes": static_row["lanes"],
        "maxspeed_kph": static_row["maxspeed_kph"],
        "base_time_min": static_row["base_time_min"],
        "hour": hour,
        "minute_bucket": minute_bucket,
        "day_of_week": day_of_week,
        "is_peak": is_peak_hour(hour),
        "oneway": static_row["oneway"],
        "bridge": static_row["bridge"],
        "tunnel": static_row["tunnel"],
        "live_factor": live_factor,
        "highway": static_row["highway"],
        "junction": static_row["junction"],
        "access": static_row["access"],
        "service": static_row["service"],
        "city": city,
    }
    return pd.DataFrame([row])


def best_edge_between(G, u, v):
    edge_dict = G.get_edge_data(u, v)
    if not edge_dict:
        return None

    best_key = None
    best_attr = None
    best_length = float("inf")

    for key, attr in edge_dict.items():
        length = float(attr.get("length", 1e9))
        if length < best_length:
            best_length = length
            best_key = key
            best_attr = attr

    if best_attr is None:
        return None
    return best_key, best_attr


def predict_edge_delay(
    model,
    u: Any,
    v: Any,
    key: Any,
    attr: Dict[str, Any],
    hour: int,
    minute_bucket: int,
    day_of_week: int,
    city: str,
    use_here: bool,
    place: str,
):
    static_row = edge_static_features(u, v, key, attr)

    live_factor = 1.0
    if use_here and callable(get_live_traffic_factor):
        try:
            live = get_live_traffic_factor(
                place=place,
                highway=static_row["highway"],
                length_m=static_row["length_m"],
                hour=hour,
                day_of_week=day_of_week,
            )
            if live is not None:
                live_factor = float(live)
        except Exception:
            live_factor = 1.0

    features = build_feature_frame(
        static_row=static_row,
        hour=hour,
        minute_bucket=minute_bucket,
        day_of_week=day_of_week,
        city=city,
        live_factor=live_factor,
    )

    predicted_delay_min = float(model.predict(features)[0])
    predicted_delay_min = max(0.0, predicted_delay_min)

    base_time_min = float(static_row["base_time_min"])
    predicted_time_min = base_time_min + predicted_delay_min

    return {
        "u": u,
        "v": v,
        "key": key,
        "highway": static_row["highway"],
        "length_m": round(static_row["length_m"], 2),
        "lanes": round(static_row["lanes"], 2),
        "maxspeed_kph": round(static_row["maxspeed_kph"], 2),
        "base_time_min": round(base_time_min, 3),
        "predicted_delay_min": round(predicted_delay_min, 3),
        "predicted_time_min": round(predicted_time_min, 3),
        "live_factor": round(live_factor, 3),
    }


def route_predictions(
    G,
    route: List[Any],
    model,
    hour: int,
    minute_bucket: int,
    day_of_week: int,
    city: str,
    use_here: bool,
    place: str,
):
    rows = []
    total_base = 0.0
    total_delay = 0.0
    total_pred = 0.0

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge = best_edge_between(G, u, v)
        if edge is None:
            continue

        key, attr = edge
        row = predict_edge_delay(
            model=model,
            u=u,
            v=v,
            key=key,
            attr=attr,
            hour=hour,
            minute_bucket=minute_bucket,
            day_of_week=day_of_week,
            city=city,
            use_here=use_here,
            place=place,
        )
        rows.append(row)
        total_base += row["base_time_min"]
        total_delay += row["predicted_delay_min"]
        total_pred += row["predicted_time_min"]

    return rows, total_base, total_delay, total_pred


with st.sidebar:
    st.header("Route settings")
    city = st.selectbox("City", CITIES, index=0)
    place = st.text_input("City / area to load graph", f"{city}, India")
    network_type = st.selectbox("Network type", ["drive", "walk", "bike"], index=0)
    origin_text = st.text_input("Origin", "Park Street, Kolkata")
    destination_text = st.text_input("Destination", "Salt Lake, Kolkata")
    use_here = st.checkbox("Use HERE live traffic hook if available", value=True)
    predict_button = st.button("Predict route", type="primary")

try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

if not predict_button:
    st.info("Choose the route settings, then click Predict route.")
    st.stop()

with st.spinner("Loading road graph..."):
    try:
        G = load_graph(place, network_type)
    except Exception as e:
        st.error(f"Could not load graph for {place}: {e}")
        st.stop()

origin = safe_geocode(origin_text)
destination = safe_geocode(destination_text)

if origin is None or destination is None:
    st.error("Could not geocode one or both locations. Try a more specific landmark or address.")
    st.stop()

try:
    origin_node = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
    destination_node = ox.distance.nearest_nodes(G, X=destination[1], Y=destination[0])
except Exception as e:
    st.error(f"Could not find nearest graph nodes: {e}")
    st.stop()

now = pd.Timestamp.now()
hour = int(now.hour)
minute_bucket = int((now.minute // 15) * 15)
day_of_week = int(now.dayofweek)

# Build a prediction-weighted graph
H = G.copy()
scored_edges = 0

for u, v, key, attr in H.edges(keys=True, data=True):
    edge_row = predict_edge_delay(
        model=model,
        u=u,
        v=v,
        key=key,
        attr=attr,
        hour=hour,
        minute_bucket=minute_bucket,
        day_of_week=day_of_week,
        city=city,
        use_here=use_here,
        place=place,
    )
    attr["predicted_delay_min"] = edge_row["predicted_delay_min"]
    attr["predicted_time_min"] = edge_row["predicted_time_min"]
    attr["weight"] = edge_row["predicted_time_min"] * 60.0  # seconds
    scored_edges += 1

try:
    route = nx.shortest_path(H, origin_node, destination_node, weight="weight")
except nx.NetworkXNoPath:
    st.error("No path found between the selected locations.")
    st.stop()

edge_rows, total_base_min, total_delay_min, total_pred_min = route_predictions(
    G=H,
    route=route,
    model=model,
    hour=hour,
    minute_bucket=minute_bucket,
    day_of_week=day_of_week,
    city=city,
    use_here=use_here,
    place=place,
)

col1, col2, col3 = st.columns(3)
col1.metric("Base travel time", f"{total_base_min:.2f} min")
col2.metric("Predicted delay", f"{total_delay_min:.2f} min")
col3.metric("Predicted trip time", f"{total_pred_min:.2f} min")

st.subheader("Route details")
st.write(f"City: **{city}**")
st.write(f"Origin node: `{origin_node}`")
st.write(f"Destination node: `{destination_node}`")
st.write(f"Edges scored: `{scored_edges}`")

st.subheader("Edge-level predictions")
pred_df = pd.DataFrame(edge_rows)
st.dataframe(pred_df, use_container_width=True, height=340)

st.subheader("Map")
try:
    fig, ax = ox.plot_graph_route(
        H,
        route,
        node_size=0,
        route_linewidth=4,
        route_alpha=0.9,
        bgcolor="white",
        show=False,
        close=False,
    )
    st.pyplot(fig, clear_figure=True)
except Exception as e:
    st.warning(f"Map rendering failed: {e}")

st.caption(
    "The model predicts delay on each road segment, injects that prediction into graph weights, "
    "and then computes the fastest route using predicted travel time."
)
