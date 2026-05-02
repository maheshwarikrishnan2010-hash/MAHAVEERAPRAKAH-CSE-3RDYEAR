import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_current_year():
    # Dynamic year or static year
    return 2026

def safe_divide(a, b):
    return a / b if b > 0 else 0

def run_feature_engineering():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    print("\n--- Starting Feature Engineering ---")
    analyzer = SentimentIntensityAnalyzer()
    
    # 1. Feature Engineering on EV Dataset
    ev_path = os.path.join(data_dir, "ev_dataset_cleaned.csv")
    if os.path.exists(ev_path):
        print("Processing EV Dataset Features...")
        try:
            # Low_memory=False is added to suppress DtypeWarning for zip_code
            ev_df = pd.read_csv(ev_path, low_memory=False)
            
            # Create vehicle_age
            if "model_year" in ev_df.columns:
                ev_df['vehicle_age'] = get_current_year() - ev_df['model_year']
                
            # Create is_high_range
            if "electric_range" in ev_df.columns:
                ev_df['is_high_range'] = ev_df['electric_range'] > 150
                
            # Drop unneeded IDs
            cols_to_drop = ["vin_1_10", "dol_vehicle_id"]
            ev_df.drop(columns=[c for c in cols_to_drop if c in ev_df.columns], inplace=True)
            
            # Save features
            out_ev_path = os.path.join(data_dir, "ev_features.csv")
            ev_df.to_csv(out_ev_path, index=False)
            print(f"Saved EV features to {out_ev_path}")
        except Exception as e:
            print(f"Error processing EV dataset: {e}")
    else:
        print(f"Cleaned EV dataset not found at {ev_path}")
        
    # 2. Feature Engineering on YouTube Dataset
    yt_path = os.path.join(data_dir, "youtube_videos_cleaned.csv")
    if os.path.exists(yt_path):
        print("Processing YouTube Dataset Features...")
        try:
            yt_df = pd.read_csv(yt_path)
            
            # NLP: Sentiment scores
            if "title" in yt_df.columns:
                yt_df["title"] = yt_df["title"].astype(str)
                yt_df['title_sentiment'] = yt_df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
                yt_df['title_length'] = yt_df['title'].apply(len)
                
            if "description" in yt_df.columns:
                yt_df["description"] = yt_df["description"].astype(str)
                yt_df['desc_sentiment'] = yt_df['description'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
                yt_df['desc_length'] = yt_df['description'].apply(len)
                
            if "tags" in yt_df.columns:
                yt_df["tags"] = yt_df["tags"].astype(str)
                yt_df['tags_count'] = yt_df['tags'].apply(lambda x: 0 if x == "Not Available" else len(x.split(',')))
                
            # Ratios
            if "likes" in yt_df.columns and "views" in yt_df.columns:
                yt_df['like_view_ratio'] = yt_df.apply(lambda row: safe_divide(row['likes'], row['views']), axis=1)
                
            if "comments_count" in yt_df.columns and "views" in yt_df.columns:
                yt_df['comment_view_ratio'] = yt_df.apply(lambda row: safe_divide(row['comments_count'], row['views']), axis=1)
                
            # Save features
            out_yt_path = os.path.join(data_dir, "youtube_features.csv")
            yt_df.to_csv(out_yt_path, index=False)
            print(f"Saved YouTube features to {out_yt_path}")
            
        except Exception as e:
            print(f"Error processing YouTube dataset: {e}")
    else:
        print(f"Cleaned YouTube dataset not found at {yt_path}")
        
    print("--- Feature Engineering Completed ---\n")

if __name__ == "__main__":
    run_feature_engineering()
