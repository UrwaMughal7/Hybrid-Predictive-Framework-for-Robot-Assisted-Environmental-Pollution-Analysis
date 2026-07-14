"""
data/loader.py
==============
Data loading functions for the Beijing Multi-Site Air Quality dataset.
  - load_hourly_data  : per-station hourly DataFrames with all 27 features
                        (11 numeric sensors + 16 one-hot wind directions)
  - load_sensor_data  : per-station hourly DataFrames with numeric sensor columns
"""

import glob
import os

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, NUMERIC_SENSORS


def load_hourly_data(data_dir: str = DATA_DIR) -> dict:
    """
    Load all 12 station CSVs → {station: hourly DataFrame with 27 features}.

    Processing (matches pm2.5.py):
      - Creates datetime index from year/month/day/hour
      - Drops No, year, month, day, hour, station
      - One-hot encodes wind direction (wd) → 16 columns
      - Fills remaining NaN with 0

    Returns dict of {station_name: DataFrame} with hourly DatetimeIndex.
    Each DataFrame has 27 columns: 11 numeric sensors + 16 one-hot wd.
    """
    print("\n" + "═" * 65)
    print("  LOADING HOURLY DATA (27 features, incl. wind direction)")
    print("═" * 65)

    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in: {data_dir}")

    out = {}
    for fp in files:
        df = pd.read_csv(fp)
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.set_index("datetime").sort_index()
        station = df["station"].iloc[0]

        # Drop metadata columns
        df = df.drop(columns=["No", "year", "month", "day", "hour", "station"])

        # One-hot encode wind direction (matches pm2.5.py)
        if "wd" in df.columns:
            df = pd.get_dummies(df, columns=["wd"])

        # Fill NaN with 0 (matches pm2.5.py)
        df = df.fillna(0)

        out[station] = df
        print(f"  ✓ {station:<22}  {len(df)} rows  {len(df.columns)} features")

    print(f"  Loaded {len(out)} stations  ({list(out.keys())[0]} … {list(out.keys())[-1]})")
    return out


def load_sensor_data(data_dir: str = DATA_DIR, features: list = None) -> dict:
    """
    Load per-station hourly data with numeric sensor columns only.
    Used by cascade simulation (no wind direction columns).

    Returns {station_name: DataFrame} with hourly DatetimeIndex.
    """
    features = features or NUMERIC_SENSORS
    hourly   = load_hourly_data(data_dir)
    out      = {}
    for sta, df in hourly.items():
        cols = [c for c in features if c in df.columns]
        out[sta] = df[cols]
    print(f"  Sensor data: {len(out)} stations, {len(features)} features")
    return out
