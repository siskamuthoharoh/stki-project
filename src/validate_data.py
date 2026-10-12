from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/produk_kecantikan.csv")
REQUIRED_COLUMNS = [
    "review_id",
    "review_text",
    "price",
    "packaging",
    "product",
    "aroma",
]
ASPECT_COLUMNS = ["price", "packaging", "product", "aroma"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SKINFIND - DATASET VALIDATION")
    print("=" * 60)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate review IDs: {df['review_id'].duplicated().sum()}")
    print(f"Duplicate review texts: {df['review_text'].duplicated().sum()}")
    print(f"Missing review texts: {df['review_text'].isna().sum()}")
    print(f"Empty review texts: {df['review_text'].fillna('').str.strip().eq('').sum()}")

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nAspect label distributions:")
    for column in ASPECT_COLUMNS:
        print(f"\n{column.upper()}")
        print(df[column].value_counts(dropna=False).sort_index().to_string())

    print("\nReview text length statistics:")
    lengths = df["review_text"].fillna("").astype(str).str.len()
    print(lengths.describe().round(2).to_string())

    print("\nSample reviews:")
    for index, review in enumerate(df["review_text"].dropna().head(3), start=1):
        print(f"{index}. {review[:250]}")

    print("\nValidation completed.")
    print("Review the results before preprocessing.")


if __name__ == "__main__":
    main()