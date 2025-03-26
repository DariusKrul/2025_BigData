# src/parallel_processing.py

import pandas as pd
from multiprocessing import Pool
from utils import haversine


def process_chunk_location(chunk, max_jump_nm=50):
    """
    Process a list of vessel DataFrames for location anomalies.
    """
    anomalies = []
    for vessel_df in chunk:
        vessel_df = vessel_df.sort_values("_timestamp")
        prev_row = None
        for _, row in vessel_df.iterrows():
            if prev_row is not None:
                dist_nm = haversine(prev_row["latitude"], prev_row["longitude"],
                                    row["latitude"], row["longitude"])
                if dist_nm > max_jump_nm:
                    anomalies.append(row)
            prev_row = row
    return anomalies


def process_chunk_speed(chunk, max_speed_knots=50):
    """
    Process a list of vessel DataFrames for speed anomalies.
    """
    anomalies = []
    for vessel_df in chunk:
        vessel_df = vessel_df.sort_values("_timestamp")
        prev_row = None
        for _, row in vessel_df.iterrows():
            if prev_row is not None:
                time_diff = (row["_timestamp"] - prev_row["_timestamp"]).total_seconds() / 3600.0
                if time_diff <= 0:
                    continue
                dist_nm = haversine(prev_row["latitude"], prev_row["longitude"],
                                    row["latitude"], row["longitude"])
                speed_knots = dist_nm / time_diff
                if speed_knots > max_speed_knots:
                    anomalies.append(row)
            prev_row = row
    return anomalies


def run_parallel_location_anomalies(df, num_workers=4, max_jump_nm=50, chunk_size=100):
    """
    Run location anomaly detection in parallel.

    Args:
        df (pd.DataFrame): Cleaned AIS data.
        num_workers (int): Number of parallel worker processes.
        max_jump_nm (float): Threshold for location jumps.
        chunk_size (int): Number of vessel groups per task.

    Returns:
        pd.DataFrame: DataFrame containing location anomalies.
    """
    groups = [group for _, group in df.groupby("mmsi")]
    # Partition groups into chunks of size `chunk_size`
    chunks = [groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)]

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_chunk_location, [(chunk, max_jump_nm) for chunk in chunks])

    anomalies = [row for sublist in results for row in sublist]
    return pd.DataFrame(anomalies)


def run_parallel_speed_anomalies(df, num_workers=4, max_speed_knots=50, chunk_size=100):
    """
    Run speed anomaly detection in parallel.

    Args:
        df (pd.DataFrame): Cleaned AIS data.
        num_workers (int): Number of parallel worker processes.
        max_speed_knots (float): Speed threshold in knots.
        chunk_size (int): Number of vessel groups per task.

    Returns:
        pd.DataFrame: DataFrame containing speed anomalies.
    """
    groups = [group for _, group in df.groupby("mmsi")]
    chunks = [groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)]

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_chunk_speed, [(chunk, max_speed_knots) for chunk in chunks])

    anomalies = [row for sublist in results for row in sublist]
    return pd.DataFrame(anomalies)
