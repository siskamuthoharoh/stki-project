
"""Preprocess beauty product reviews for the SkinFind project."""

from pathlib import Path

import pandas as pd

from src.text_preprocessing import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "produk_kecantikan.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "produk_kecantikan_bersih.csv"
)


def main() -> None:
    """Load, preprocess, validate, and save the review dataset."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"review_id", "review_text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}"
        )

    original_count = len(df)

    # Preserve original review text and create a cleaned text column.
    df["cleaned_text"] = df["review_text"].map(preprocess_text)

    # Ensure all original records remain in the output.
    if len(df) != original_count:
        raise ValueError("Jumlah baris berubah selama prapemrosesan.")

    # Check for empty results from non-empty original reviews.
    original_nonempty = df["review_text"].fillna("").str.strip().ne("")
    cleaned_empty = df["cleaned_text"].eq("")
    empty_after_processing = int((original_nonempty & cleaned_empty).sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("PEMROSESAN DATASET SELESAI")
    print("=" * 45)
    print(f"Jumlah ulasan awal       : {original_count:,}")
    print(f"Jumlah ulasan hasil      : {len(df):,}")
    print(f"Kolom hasil              : {len(df.columns)}")
    print(f"Hasil kosong dari teks asli nonkosong: {empty_after_processing}")
    print(f"File tersimpan           : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
