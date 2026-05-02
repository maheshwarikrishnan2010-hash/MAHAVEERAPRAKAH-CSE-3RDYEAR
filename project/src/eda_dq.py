import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as plt_sns
import sys

# use seaborn styles
plt_sns.set_theme(style="whitegrid")

def generate_ev_report(df, report_file, visuals_dir):
    report_file.write("--- EV Dataset Data Quality & EDA ---\n")
    report_file.write(f"Total rows: {len(df)}\n")
    report_file.write(f"Total columns: {len(df.columns)}\n\n")
    
    # Missing Values
    report_file.write("Missing Values:\n")
    report_file.write(df.isnull().sum()[df.isnull().sum() > 0].to_string() + "\n\n")
    
    # Duplicates
    report_file.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")

    # EDA
    if "make" in df.columns:
        top_makes = df["make"].value_counts().head(10)
        report_file.write("Top 10 EV Makes:\n")
        report_file.write(top_makes.to_string() + "\n\n")
        
        # Plot
        plt.figure(figsize=(10,6))
        plt_sns.barplot(x=top_makes.values, y=top_makes.index, hue=top_makes.index, palette="viridis", legend=False)
        plt.title("Top 10 EV Makes")
        plt.xlabel("Count")
        plt.ylabel("Make")
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, "ev_top_makes.png"))
        plt.close()
        
    if "electric_range" in df.columns:
        report_file.write("Electric Range Statistics:\n")
        report_file.write(df["electric_range"].describe().to_string() + "\n\n")
        
        plt.figure(figsize=(8,5))
        plt_sns.histplot(df[df["electric_range"] > 0]["electric_range"], bins=30, kde=True, color="blue")
        plt.title("Electric Range Distribution (excluding 0 range)")
        plt.xlabel("Electric Range (miles)")
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, "ev_electric_range_dist.png"))
        plt.close()

def generate_youtube_report(df, report_file, visuals_dir):
    report_file.write("--- YouTube Videos Data Quality & EDA ---\n")
    report_file.write(f"Total rows: {len(df)}\n")
    report_file.write(f"Total columns: {len(df.columns)}\n\n")
    
    # Missing values
    report_file.write("Missing Values:\n")
    report_file.write(df.isnull().sum()[df.isnull().sum() > 0].to_string() + "\n\n")
    
    # Duplicates (check by video_id)
    if "video_id" in df.columns:
        dups = df.duplicated(subset=["video_id"]).sum()
        report_file.write(f"Duplicate video_ids: {dups}\n\n")
    else:
        report_file.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")

    # EDA
    if "views" in df.columns and "likes" in df.columns:
        report_file.write("Engagement Statistics:\n")
        report_file.write(df[["views", "likes", "comments_count"]].describe().to_string() + "\n\n")
        
        # Scatter Plot Views vs Likes
        plt.figure(figsize=(8,5))
        plt_sns.scatterplot(data=df, x="views", y="likes", alpha=0.6, color="red")
        plt.title("Views vs Likes for EV Videos")
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, "youtube_views_vs_likes.png"))
        plt.close()

def run_eda_dq():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    visuals_dir = os.path.join(reports_dir, "visuals")
    
    os.makedirs(visuals_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, "data_quality_report.txt")
    print(f"Generating consolidated report at {report_path}...")
    
    with open(report_path, "w") as f:
        f.write("=== CONSOLIDATED EDA & DATA QUALITY REPORT ===\n\n")
        
        # EV Dataset
        ev_path = os.path.join(data_dir, "ev_dataset.csv")
        if os.path.exists(ev_path):
            try:
                ev_df = pd.read_csv(ev_path)
                generate_ev_report(ev_df, f, visuals_dir)
            except Exception as e:
                f.write(f"Error processing EV Dataset: {e}\n\n")
        else:
            f.write("EV Dataset not found.\n\n")
            
        f.write("-" * 40 + "\n\n")
        
        # YouTube Videos
        yt_path = os.path.join(data_dir, "youtube_videos.csv")
        if os.path.exists(yt_path):
            try:
                yt_df = pd.read_csv(yt_path)
                generate_youtube_report(yt_df, f, visuals_dir)
            except Exception as e:
                f.write(f"Error processing YouTube Dataset: {e}\n\n")
        else:
            f.write("YouTube Dataset not found.\n\n")
            
    print("Done! Visuals saved to", visuals_dir)

if __name__ == "__main__":
    run_eda_dq()
