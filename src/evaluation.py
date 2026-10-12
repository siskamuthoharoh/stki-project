
"""Evaluate SkinFind search results using manually labeled relevance."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "kandidat_evaluasi_diperbaiki.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hasil_evaluasi.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "ringkasan_evaluasi.csv"


def average_precision(relevance):
    """Calculate Average Precision for one ranked query."""
    relevance = list(relevance)
    relevant_found = 0
    precision_sum = 0.0

    for rank, label in enumerate(relevance, start=1):
        if label == 1:
            relevant_found += 1
            precision_sum += relevant_found / rank

    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    return precision_sum / total_relevant


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File kandidat tidak ditemukan: {INPUT_PATH}"
        )

    # File yang diedit di Excel menggunakan pemisah titik koma.
    data = pd.read_csv(INPUT_PATH, sep=";", decimal=",")

    required_columns = {"query", "rank", "relevant"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}"
        )

    data["relevant"] = pd.to_numeric(data["relevant"], errors="coerce")
    data["rank"] = pd.to_numeric(data["rank"], errors="coerce")

    if data["relevant"].isna().any():
        raise ValueError("Masih ada label relevant yang kosong atau tidak valid.")

    if not data["relevant"].isin([0, 1]).all():
        raise ValueError("Kolom relevant hanya boleh berisi 0 atau 1.")

    if data["rank"].isna().any():
        raise ValueError("Ada nilai rank yang tidak valid.")

    data = data.sort_values(["query", "rank"]).copy()
    results = []

    for query, group in data.groupby("query", sort=False):
        labels = group["relevant"].astype(int).tolist()

        precision_at_5 = sum(labels[:5]) / min(5, len(labels))
        precision_at_10 = sum(labels[:10]) / min(10, len(labels))
        ap = average_precision(labels)

        results.append(
            {
                "query": query,
                "jumlah_kandidat": len(labels),
                "jumlah_relevan": sum(labels),
                "precision_at_5": precision_at_5,
                "precision_at_10": precision_at_10,
                "average_precision": ap,
            }
        )

    summary = pd.DataFrame(results)

    if summary.empty:
        raise ValueError("Tidak ada kueri yang dapat dievaluasi.")

    map_score = summary["average_precision"].mean()
    summary["MAP_seluruh_kueri"] = map_score

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("=== HASIL EVALUASI SKINFIND ===")
    print(f"Jumlah kandidat: {len(data)}")
    print(f"Jumlah kueri: {data['query'].nunique()}")
    print(f"Label relevan: {(data['relevant'] == 1).sum()}")
    print(f"Label tidak relevan: {(data['relevant'] == 0).sum()}")
    print(f"MAP keseluruhan: {map_score:.4f}")
    print("\nRincian per kueri:")
    print(summary.to_string(index=False))
    print(f"\nHasil tersimpan: {OUTPUT_PATH}")
    print(f"Ringkasan tersimpan: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
