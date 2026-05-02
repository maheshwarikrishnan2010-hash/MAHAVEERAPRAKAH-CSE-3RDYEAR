from sklearn.model_selection import train_test_split

def split_data(df):

    features = [
        "model_year",
        "make",
        "model",
        "city",
        "county",
        "vehicle_age",
        "ev_count"
    ]

    # Handle missing columns by only selecting what's available
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]

    # Handle missing target columns
    if "electric_range" in df.columns:
        y_reg = df["electric_range"]
    else:
        y_reg = [0] * len(df) # Placeholder
        print("Warning: electric_range column missing")
        
    if "EV_Adoption_Level" in df.columns:
        y_clf = df["EV_Adoption_Level"]
    elif "is_high_range" in df.columns:
        y_clf = df["is_high_range"]
    else:
        y_clf = [0] * len(df) # Placeholder
        print("Warning: EV_Adoption_Level and is_high_range columns missing")

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    X_train_c, X_test_c, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train_reg, y_test_reg, X_train_c, X_test_c, y_train_clf, y_test_clf
