# src/preprocess.py

import pandas as pd


def load_data(filepath, chunksize=500_000):
    """
    Load AIS data in chunks and concatenate into a full DataFrame.
    """
    print(f"Loading data from {filepath} in chunks...")
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Normalize column names: lowercase, strip spaces, remove '#' if present
        chunk.columns = (
            chunk.columns.str.lower()
            .str.strip()
            .str.replace("#", "")
            .str.replace(" ", "_")
        )
        chunks.append(chunk)
        print(f"Loaded chunk with {len(chunk)} rows")

    df = pd.concat(chunks, ignore_index=True)
    print("Columns in loaded dataframe:", df.columns.tolist())

    print(f"Total records loaded: {len(df):,}")
    return df


def clean_data(df):
    """
    Clean the AIS data by dropping invalid entries and converting types.

    Expected normalized columns:
    - mmsi
    - _timestamp
    - latitude
    - longitude
    - sog
    - cog
    """
    required_cols = ['mmsi', '_timestamp', 'latitude', 'longitude', 'sog', 'cog']
    df = df.dropna(subset=required_cols)

    # Convert _timestamp to datetime
    df.loc[:, '_timestamp'] = pd.to_datetime(df['_timestamp'], errors='coerce')
    df = df.dropna(subset=['_timestamp'])

    # Convert numeric columns
    df.loc[:, 'latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df.loc[:, 'longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df.loc[:, 'sog'] = pd.to_numeric(df['sog'], errors='coerce')
    df.loc[:, 'cog'] = pd.to_numeric(df['cog'], errors='coerce')

    df = df.dropna(subset=['latitude', 'longitude', 'sog', 'cog'])

    # Sort by vessel (mmsi) and time
    df = df.sort_values(by=['mmsi', '_timestamp']).reset_index(drop=True)

    return df


