# src/spoofing_detection.py

import pandas as pd
from datetime import timedelta
from utils import haversine


def detect_location_anomalies(df, max_jump_nm=50):
    """
    Detects sudden location jumps that exceed a fixed maximum jump threshold.

    Args:
        df (pd.DataFrame): Cleaned AIS data.
        max_jump_nm (float): Maximum allowed jump in nautical miles.

    Returns:
        pd.DataFrame: A DataFrame of records flagged as location anomalies.
    """
    anomalies = []
    for mmsi, vessel_df in df.groupby("mmsi"):
        vessel_df = vessel_df.sort_values("_timestamp")
        prev_row = None
        for _, row in vessel_df.iterrows():
            if prev_row is not None:
                dist_nm = haversine(
                    prev_row["latitude"],
                    prev_row["longitude"],
                    row["latitude"],
                    row["longitude"],
                )
                if dist_nm > max_jump_nm:
                    anomalies.append(row)
            prev_row = row
    return pd.DataFrame(anomalies)


def detect_speed_anomalies(df, max_speed_knots=50):
    """
    Detects impossible speed jumps between consecutive points.

    Args:
        df (pd.DataFrame): Cleaned AIS data.
        max_speed_knots (float): Maximum allowed vessel speed in knots.

    Returns:
        pd.DataFrame: A DataFrame of records flagged as speed anomalies.
    """
    anomalies = []
    for mmsi, vessel_df in df.groupby("mmsi"):
        vessel_df = vessel_df.sort_values("_timestamp")
        prev_row = None
        for _, row in vessel_df.iterrows():
            if prev_row is not None:
                time_diff = (row["_timestamp"] - prev_row["_timestamp"]).total_seconds() / 3600.0
                if time_diff <= 0:
                    continue
                dist_nm = haversine(
                    prev_row["latitude"],
                    prev_row["longitude"],
                    row["latitude"],
                    row["longitude"],
                )
                speed_knots = dist_nm / time_diff
                if speed_knots > max_speed_knots:
                    anomalies.append(row)
            prev_row = row
    return pd.DataFrame(anomalies)
