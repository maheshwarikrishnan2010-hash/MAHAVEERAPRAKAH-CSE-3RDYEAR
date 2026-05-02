import pandas as pd
from .ev_api import fetch_ev_data
from .youtube_api import fetch_videos, fetch_comments
from .storage import save_data
from .format_selector import choose_format
from .data_cleaning import run_data_cleaning
from .feature_eng import run_feature_engineering
import os

from .preprocessing.encoding import encode_data
from .preprocessing.scaling import scale_data
from .modeling.data_split import split_data
from .modeling.model_runner import run_regression, run_classification

def run_models(df):

    df = encode_data(df)

    X_train, X_test, y_train_reg, y_test_reg, X_train_c, X_test_c, y_train_clf, y_test_clf = split_data(df)

    # Note: scale_data could be applied to X_train and X_test if required here. 
    # For now, keeping structure exactly as requested.

    reg_results = run_regression(X_train, y_train_reg)
    clf_results = run_classification(X_train_c, y_train_clf)

    print("\nRegression Results:\n", reg_results)
    print("\nClassification Results:\n", clf_results)
    
    return reg_results, clf_results

def run_pipeline():

    fmt = choose_format()

    # ev data
    ev_df = fetch_ev_data()
    save_data(ev_df, "ev_dataset", fmt)

    # youtube videos
    topics = ["Tesla review", "Nissan Leaf review", "EV comparison", "electric car range"]
    all_videos_dfs = []

    for topic in topics:
        try:
            videos_df = fetch_videos(query=topic)
            all_videos_dfs.append(videos_df)
        except Exception as e:
            print(f"Error fetching for {topic}: {e}")
            continue

    if all_videos_dfs:
        combined_videos_df = pd.concat(all_videos_dfs, ignore_index=True)
        if "video_id" in combined_videos_df.columns:
            combined_videos_df.drop_duplicates(subset=["video_id"], inplace=True)
    else:
        combined_videos_df = pd.DataFrame()

    save_data(combined_videos_df, "youtube_videos", fmt)

    # youtube comments
    all_comments = []

    if "video_id" in combined_videos_df.columns:
        for vid in combined_videos_df["video_id"]:
    
            try:
                comments = fetch_comments(vid)
                all_comments.extend(comments)
            except:
                continue

    comments_df = pd.DataFrame(all_comments)

    save_data(comments_df, "youtube_comments", fmt)

    print("\n Data Injection Completed")
    run_data_cleaning()
    run_feature_engineering()
    
    # Load feature engineered data for modeling
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    ev_features_path = os.path.join(data_dir, "ev_features.csv")
    
    if os.path.exists(ev_features_path):
        print("\n--- Starting Modeling ---")
        ev_df = pd.read_csv(ev_features_path, low_memory=False)
        run_models(ev_df)
        print("--- Modeling Completed ---\n")
    else:
        print("\nWarning: ev_features.csv not found. Skipping modeling phase.\n")