from sklearn.preprocessing import LabelEncoder

def encode_data(df):

    df = df.copy()

    categorical_cols = ["make", "model", "city", "county"]

    le = LabelEncoder()

    for col in categorical_cols:

        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    return df
