"""Download the UCI Heart Disease dataset and save a cleaned CSV."""
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def main():
    heart = fetch_ucirepo(id=45)  # Heart Disease dataset
    X = heart.data.features
    y = heart.data.targets

    df = pd.concat([X, y], axis=1)
    df = df.rename(columns={"num": "target"})
    df.to_csv(DATA_DIR / "heart_raw.csv", index=False)

    # Binarize target: 0 = no disease, 1 = disease (original values 1-4)
    df["target"] = (df["target"] > 0).astype(int)

    # Missing values appear in 'ca' and 'thal'
    print("Missing values:\n", df.isna().sum())
    df = df.fillna(df.median(numeric_only=True))

    df.to_csv(DATA_DIR / "heart_clean.csv", index=False)
    print(f"Saved {len(df)} rows to data/heart_clean.csv")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()
