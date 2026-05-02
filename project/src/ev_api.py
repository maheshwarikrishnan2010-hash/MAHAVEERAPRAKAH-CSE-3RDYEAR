import requests
import pandas as pd
import os
from .config import EV_API_URL

def fetch_ev_data():

    print("Connecting to EV dataset API...")

    try:
        response = requests.get(EV_API_URL, params={"$limit": 1000000}, timeout=10)

        if response.status_code != 200:
            raise Exception("Failed to fetch data from API")

        data = response.json()

        df = pd.DataFrame(data)

        print("Dataset downloaded successfully")

        return df
    except Exception as e:
        print(f"API Connection Failed: {e}")
        print("Falling back to local cached dataset (data/ev_dataset.csv)...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(base_dir, "data", "ev_dataset.csv")
        
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, low_memory=False)
            print("Local dataset loaded successfully.")
            return df
        else:
            raise Exception("API failed and local dataset 'ev_dataset.csv' not found.")