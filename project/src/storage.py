import os

def save_data(df, filename_base, fmt):

    os.makedirs("data", exist_ok=True)

    path = f"data/{filename_base}.{fmt}"

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", indent=4)
    elif fmt == "xlsx":
        df.to_excel(path, index=False)
    else:
        path = f"data/{filename_base}.csv"
        df.to_csv(path, index=False)

    print(f"Saved → {path}")