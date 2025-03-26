import sys
import os
import time

# Add the src folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import preprocess
import spoofing_detection
import parallel_processing


def main():
    filepath = "data/ais_data.csv"  # Replace with your actual file path
    df = preprocess.load_data(filepath)
    df_clean = preprocess.clean_data(df)
    print(f"Cleaned data shape: {df_clean.shape}")

    # --- Sequential execution (commented out to save time) ---
    """
    start_seq = time.time()
    loc_anoms_seq = spoofing_detection.detect_location_anomalies(df_clean)
    speed_anoms_seq = spoofing_detection.detect_speed_anomalies(df_clean)
    end_seq = time.time()
    print(f"Sequential location anomalies: {len(loc_anoms_seq)}")
    print(f"Sequential speed anomalies: {len(speed_anoms_seq)}")
    print(f"Sequential processing time: {end_seq - start_seq:.2f} seconds")
    """

    # --- Parallel execution ---
    start_par = time.time()
    loc_anoms_par = parallel_processing.run_parallel_location_anomalies(df_clean, num_workers=4)
    speed_anoms_par = parallel_processing.run_parallel_speed_anomalies(df_clean, num_workers=4)
    end_par = time.time()
    print(f"Parallel location anomalies: {len(loc_anoms_par)}")
    print(f"Parallel speed anomalies: {len(speed_anoms_par)}")
    print(f"Parallel processing time: {end_par - start_par:.2f} seconds")

    # Save the parallel results to CSV files
    os.makedirs("results", exist_ok=True)
    loc_anoms_par.to_csv("results/location_anomalies_parallel.csv", index=False)
    speed_anoms_par.to_csv("results/speed_anomalies_parallel.csv", index=False)
    print("Parallel anomaly results saved to 'results/' folder.")


if __name__ == "__main__":
    main()
