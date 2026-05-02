import pandas as pd
import os

def run_data_cleaning():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    print("\n--- Starting Data Cleaning ---")
    
    # 1. Clean EV Dataset
    ev_path = os.path.join(data_dir, "ev_dataset.csv")
    if os.path.exists(ev_path):
        print("Cleaning EV Dataset...")
        try:
            ev_df = pd.read_csv(ev_path)
            
            # Drop unnecessary columns
            cols_to_drop = [
                '_2020_census_tract', 
                'legislative_district', 
                'geocoded_column', 
                ':@computed_region_x4ys_rtnd', 
                ':@computed_region_fny7_vc3j', 
                ':@computed_region_8ddd_yn5v'
            ]
            ev_df.drop(columns=[c for c in cols_to_drop if c in ev_df.columns], inplace=True)
            
            # Fill missing categorical values
            cat_cols = ['county', 'city', 'zip_code', 'electric_utility']
            for col in cat_cols:
                if col in ev_df.columns:
                    ev_df[col] = ev_df[col].fillna('Unknown')
            
            # Fill missing numeric values
            if 'electric_range' in ev_df.columns:
                ev_df['electric_range'] = ev_df['electric_range'].fillna(0)
                
            # Save cleaned dataset
            cleaned_ev_path = os.path.join(data_dir, "ev_dataset_cleaned.csv")
            ev_df.to_csv(cleaned_ev_path, index=False)
            print(f"Saved cleaned EV dataset to {cleaned_ev_path}")
        except Exception as e:
            print(f"Error cleaning EV dataset: {e}")
    else:
        print(f"EV Dataset not found at {ev_path}")
        
    # 2. Clean YouTube Videos Dataset
    yt_path = os.path.join(data_dir, "youtube_videos.csv")
    if os.path.exists(yt_path):
        print("Cleaning YouTube Videos Dataset...")
        try:
            yt_df = pd.read_csv(yt_path)
            
            # Fill missing text
            text_cols = ['description', 'tags']
            for col in text_cols:
                if col in yt_df.columns:
                    yt_df[col] = yt_df[col].fillna("Not Available")
            
            # Fill missing metrics
            metric_cols = ['likes', 'comments_count']
            for col in metric_cols:
                if col in yt_df.columns:
                    yt_df[col] = yt_df[col].fillna(0)
                    
            # Save cleaned dataset
            cleaned_yt_path = os.path.join(data_dir, "youtube_videos_cleaned.csv")
            yt_df.to_csv(cleaned_yt_path, index=False)
            print(f"Saved cleaned YouTube data to {cleaned_yt_path}")
        except Exception as e:
            print(f"Error cleaning YouTube dataset: {e}")
    else:
        print(f"YouTube Videos dataset not found at {yt_path}")
        
    print("--- Data Cleaning Completed ---\n")

if __name__ == "__main__":
    run_data_cleaning()
