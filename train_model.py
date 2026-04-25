"""
train_model.py

Trains an XGBoost model for road-segment delay prediction.

Data sources:
- OSM-derived synthetic/historical road dataset from osm_data.py
- Manual multi-city observations from manual_google_data_multi_city.csv

Outputs:
- model/delay_model.pkl
- model/metrics.json
- model/feature_columns.json

Target:
- delay_min = actual travel time - baseline travel time
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

OSM_DATA_PATH = DATA_DIR / "road_delay_dataset.csv"
MANUAL_DATA_PATH = Path("manual_google_data_multi_city.csv")

MODEL_PATH = MODEL_DIR / "delay_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

NUMERIC_FEATURES = [
    "length_m",
    "lanes",
    "maxspeed_kph",
    "base_time_min",
    "hour",
    "minute_bucket",
    "day_of_week",
    "is_peak",
    "oneway",
    "bridge",
    "tunnel",
    "live_factor",
]

CATEGORICAL_FEATURES = [
    "highway",
    "junction",
    "access",
    "service",
    "city",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "delay_min"


def _ensure_columns(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    out = df.copy()
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    return out


def _standardize_osm_df(df: pd.DataFrame, default_city: str = "Kolkata") -> pd.DataFrame:
    """
    Standardize the OSM-derived dataset into the feature space used by the model.
    """
    df = df.copy()

    # Support older/newer versions of osm_data.py
    if "day" in df.columns and "day_of_week" not in df.columns:
        df["day_of_week"] = df["day"]
    if "delay" in df.columns and TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = df["delay"]
    if "observed_time_min" in df.columns and TARGET_COLUMN not in df.columns and "base_time_min" in df.columns:
        df[TARGET_COLUMN] = df["observed_time_min"] - df["base_time_min"]

    df = _ensure_columns(
        df,
        {
            "city": default_city,
            "minute_bucket": 0,
            "is_peak": 0,
            "live_factor": 1.0,
            "oneway": 0,
            "bridge": 0,
            "tunnel": 0,
            "junction": "none",
            "access": "unknown",
            "service": "none",
            "highway": "other",
            "lanes": 1.0,
            "maxspeed_kph": 30.0,
        },
    )

    # Make sure types are sane
    df["city"] = df["city"].astype(str)
    df["highway"] = df["highway"].astype(str)
    df["junction"] = df["junction"].astype(str)
    df["access"] = df["access"].astype(str)
    df["service"] = df["service"].astype(str)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"OSM dataset must contain {TARGET_COLUMN} or delay/observed_time_min.")

    required = ["length_m", "lanes", "maxspeed_kph", "base_time_min", "hour", "minute_bucket", "day_of_week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OSM dataset is missing required columns: {missing}")

    return df


def _city_profile(city: str) -> dict:
    """
    Reasonable fallback assumptions for the manual Google Maps style dataset.
    These are only used to convert the human-collected observations into the
    same feature shape as the OSM dataset.
    """
    c = str(city).strip().lower()
    profiles = {
        "kolkata": {"lanes": 2.0, "maxspeed_kph": 28.0, "highway": "primary"},
        "bangalore": {"lanes": 3.0, "maxspeed_kph": 32.0, "highway": "primary"},
        "hyderabad": {"lanes": 3.0, "maxspeed_kph": 35.0, "highway": "secondary"},
        "delhi": {"lanes": 3.0, "maxspeed_kph": 38.0, "highway": "primary"},
        "mumbai": {"lanes": 2.0, "maxspeed_kph": 30.0, "highway": "secondary"},
    }
    return profiles.get(c, {"lanes": 2.0, "maxspeed_kph": 30.0, "highway": "primary"})


def _standardize_manual_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the manual Google Maps style CSV into the same feature space.
    Expected columns:
        route_id, city, start, end, hour, day, expected_time, actual_time
    """
    if df.empty:
        return df

    df = df.copy()

    rename_map = {}
    if "day" in df.columns and "day_of_week" not in df.columns:
        rename_map["day"] = "day_of_week"
    if "expected_time" in df.columns and "base_time_min" not in df.columns:
        rename_map["expected_time"] = "base_time_min"
    if "actual_time" in df.columns and "observed_time_min" not in df.columns:
        rename_map["actual_time"] = "observed_time_min"
    df = df.rename(columns=rename_map)

    required = ["city", "hour", "day_of_week", "base_time_min", "observed_time_min"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manual dataset is missing required columns: {missing}")

    # Convert to the model target
    df[TARGET_COLUMN] = df["observed_time_min"].astype(float) - df["base_time_min"].astype(float)

    # City-aware proxies for road attributes
    cities = []
    lanes = []
    maxspeeds = []
    highways = []
    lengths = []

    for _, row in df.iterrows():
        city = str(row["city"])
        profile = _city_profile(city)
        base_time = float(row["base_time_min"])

        # Approximate road length from the baseline travel time and typical city speed.
        # length_m = time(hours) * speed(km/h) * 1000
        approx_length_m = max(250.0, (base_time / 60.0) * profile["maxspeed_kph"] * 1000.0 * 0.9)

        cities.append(city)
        lanes.append(profile["lanes"])
        maxspeeds.append(profile["maxspeed_kph"])
        highways.append(profile["highway"])
        lengths.append(approx_length_m)

    df["city"] = cities
    df["lanes"] = lanes
    df["maxspeed_kph"] = maxspeeds
    df["highway"] = highways
    df["length_m"] = lengths

    df["minute_bucket"] = 0
    df["is_peak"] = df["hour"].apply(lambda h: 1 if (7 <= int(h) <= 10 or 17 <= int(h) <= 20) else 0)
    df["live_factor"] = 1.0
    df["oneway"] = 0
    df["bridge"] = 0
    df["tunnel"] = 0
    df["junction"] = "none"
    df["access"] = "yes"
    df["service"] = "none"

    # Keep route metadata if present, but don't use it as a feature
    for col in ["route_id", "start", "end"]:
        if col not in df.columns:
            df[col] = ""

    return df


def load_datasets(osm_city: str = "Kolkata") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not OSM_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{OSM_DATA_PATH} not found. Run osm_data.py first to build the OSM dataset."
        )

    osm_df = pd.read_csv(OSM_DATA_PATH)
    osm_df = _standardize_osm_df(osm_df, default_city=osm_city)

    manual_df = pd.DataFrame()
    if MANUAL_DATA_PATH.exists():
        manual_df = pd.read_csv(MANUAL_DATA_PATH)
        if len(manual_df) > 0:
            manual_df = _standardize_manual_df(manual_df)

    if len(manual_df) > 0:
        combined = pd.concat([osm_df, manual_df], ignore_index=True, sort=False)
    else:
        combined = osm_df.copy()

    # Add sample weights so real/manual observations matter more
    combined["sample_weight"] = 1.0
    if "route_id" in combined.columns:
        manual_mask = combined["route_id"].fillna("").astype(str).str.len() > 0
        combined.loc[manual_mask, "sample_weight"] = 3.0

    # Ensure all model columns are present
    combined = _ensure_columns(
        combined,
        {
            "city": osm_city,
            "minute_bucket": 0,
            "is_peak": 0,
            "live_factor": 1.0,
            "oneway": 0,
            "bridge": 0,
            "tunnel": 0,
            "junction": "none",
            "access": "unknown",
            "service": "none",
        },
    )

    return osm_df, manual_df, combined


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train(df: pd.DataFrame):
    df = df.copy()

    # Keep only rows that have all required columns
    required = FEATURE_COLUMNS + [TARGET_COLUMN, "sample_weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Combined dataset is missing columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    weights = df["sample_weight"].astype(float).copy()

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train, model__sample_weight=w_train)

    preds = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {
        "rmse_min": rmse,
        "mae_min": mae,
        "r2": r2,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(df)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "manual_rows": int((df["sample_weight"] > 1.0).sum()),
        "osm_rows": int((df["sample_weight"] <= 1.0).sum()),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return pipe, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the road delay prediction model.")
    parser.add_argument("--osm_city", type=str, default="Kolkata", help="City label for the OSM dataset if it has no city column.")
    args = parser.parse_args()

    osm_df, manual_df, combined = load_datasets(osm_city=args.osm_city)
    model, metrics = train(combined)

    print(f"OSM rows: {len(osm_df):,}")
    print(f"Manual rows: {len(manual_df):,}")
    print(f"Combined rows: {len(combined):,}")
    print(f"Saved model to: {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
