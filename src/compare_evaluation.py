
"""Compare baseline and updated SkinFind ranking on the same judged pool."""

from pathlib import Path

import pandas as pd

from src.retrieval import SkinFindSearch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "kandidat_evaluasi_diperbaiki.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "hasil_evaluasi_boost.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "ringkasan_evaluasi_boost.csv"
)


def average_precision(relevance):
    """Calculate Average Precision for a ranked candidate pool."""
    relevant_found = 0
    precision_sum = 0.0

    for rank, label in enumerate(relevance, start=1):
        if label == 1:
            relevant_found += 1
            precision_sum += relevant_found / rank

    total_relevant = sum(relevance)
    return precision_sum / total_relevant if total_relevant else 0.0


def calculate_metrics(data, rank_column):
    """Calculate ranking metrics for each query."""
    rows = []

    for query, group in data.groupby("query", sort=False):
        group = group.sort_values(rank_column)
        labels = group["relevant"].astype(int).tolist()

        rows.append(
            {
                "query": query,
                "jumlah_kandidat": len(labels),
                "jumlah_relevan": sum(labels),
                "precision_at_5": sum(labels[:5]) / min(5, len(labels)),
                "precision_at_10": sum(labels[:10]) / min(10, len(labels)),
                "average_precision": average_precision(labels),
            }
        )

    return pd.DataFrame(rows)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File kandidat tidak ditemukan: {INPUT_PATH}")

    # Read the manually labeled file without changing its labels.
    data = pd.read_csv(
        INPUT_PATH, sep=";", decimal=",", encoding="utf-8-sig"
    )

    required = {"query", "rank", "review_id", "relevant"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    data["rank"] = pd.to_numeric(data["rank"], errors="raise")
    data["review_id"] = pd.to_numeric(data["review_id"], errors="raise")
    data["relevant"] = pd.to_numeric(data["relevant"], errors="raise")

    if data["relevant"].isna().any() or not data["relevant"].isin([0, 1]).all():
        raise ValueError("Label relevant harus terisi dan hanya berisi 0 atau 1.")

    if data.duplicated(["query", "review_id"]).any():
        raise ValueError("Ada review_id ganda dalam kueri yang sama.")

    # Baseline uses the original saved rank order.
    baseline = calculate_metrics(data, "rank")
    baseline_map = baseline["average_precision"].mean()

    # Re-score the same judged candidates using the current retrieval model.
    searcher = SkinFindSearch()
    updated = data.copy()
    updated["new_similarity_score"] = 0.0

    for query in updated["query"].unique():
        candidate_mask = updated["query"] == query
        candidate_ids = updated.loc[candidate_mask, "review_id"]

        results = searcher.search(query, top_k=len(searcher.df))
        score_lookup = results.set_index("review_id")["similarity_score"]

        updated.loc[candidate_mask, "new_similarity_score"] = (
            candidate_ids.map(score_lookup).fillna(0.0).to_numpy()
        )

    # Keep the same judged pool and labels; only update the ranking.
    updated = updated.sort_values(
        ["query", "new_similarity_score", "review_id"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()

    updated["new_rank"] = updated.groupby("query").cumcount() + 1
    updated_metrics = calculate_metrics(updated, "new_rank")
    updated_map = updated_metrics["average_precision"].mean()

    comparison = baseline.merge(
        updated_metrics,
        on="query",
        suffixes=("_baseline", "_boost"),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    comparison.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("=== PERBANDINGAN RANKING SKINFIND ===")
    print(f"Jumlah kandidat yang sama: {len(data)}")
    print(f"Jumlah kueri: {data['query'].nunique()}")
    print(f"Label relevan: {(data['relevant'] == 1).sum()}")
    print(f"MAP baseline: {baseline_map:.4f}")
    print(f"MAP ranking terbaru: {updated_map:.4f}")
    print(f"Perubahan MAP: {updated_map - baseline_map:+.4f}")
    print("\nPerbandingan per kueri:")
    print(comparison.to_string(index=False))
    print(f"\nHasil ranking tersimpan: {OUTPUT_PATH}")
    print(f"Ringkasan tersimpan: {SUMMARY_PATH}")
    print("\nCatatan: evaluasi ini memakai kumpulan kandidat berlabel yang sama.")


if __name__ == "__main__":
    main()
