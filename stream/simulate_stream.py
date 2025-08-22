# stream/simulate_stream.py

import pandas as pd
import requests
import time
import json
import os

print("Starting data stream simulation...")

# --- 1. DEFINE PATHS ---
# Get the directory where the current script is located
script_dir = os.path.dirname(__file__) 
# Build the absolute path to the data file
DATA_PATH = os.path.normpath(os.path.join(script_dir, '..', 'data', 'CICIDS_sample.csv'))
# *** IMPORTANT: Use the new API endpoint ***
API_ENDPOINT = "http://127.0.0.1:5000/api/detect"

# --- 2. LOAD DATA ---
try:
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# --- 3. CLEAN COLUMN NAMES ---
df.columns = df.columns.str.strip()


# --- 4. START STREAMING ---
print(f"Streaming {len(df)} records to {API_ENDPOINT}...")

# Loop over the entire DataFrame
for index, row in df.iterrows():
    try:
        # Convert row to a dictionary
        log_data = row.to_dict()
        
        # Send data to the Flask API
        response = requests.post(API_ENDPOINT, json=log_data)
        
        # Print status
        if response.status_code == 200:
            print(f"Record {index + 1}: Sent successfully. Status: {response.json().get('status')}")
        else:
            print(f"Record {index + 1}: Failed to send. Status Code: {response.status_code}, Response: {response.text}")

        # Simulate real-time delay
        time.sleep(0.5) # Send a record every 0.5 seconds

    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Could not connect to the server at {API_ENDPOINT}.")
        print("Please ensure the Flask app (app.py) is running.")
        break
    except Exception as e:
        print(f"An error occurred at record {index + 1}: {e}")
        continue

print("Stream simulation finished.")
