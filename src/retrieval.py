
"""TF-IDF-based information retrieval for SkinFind."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_preprocessing import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "produk_kecantikan_bersih.csv"
)


class SkinFindSearch:
    """Search skincare reviews using TF-IDF and cosine similarity."""

    def __init__(self, data_path: Path | str = DATA_PATH) -> None:
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {self.data_path}"
            )

        self.df = pd.read_csv(self.data_path)

        required_columns = {"review_id", "review_text", "cleaned_text"}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}"
            )

        self.df["cleaned_text"] = self.df["cleaned_text"].fillna("").astype(str)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["cleaned_text"]
        )

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Return the most relevant reviews for a search query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Kata kunci pencarian tidak boleh kosong.")

        if top_k < 1:
            raise ValueError("top_k harus minimal 1.")

        cleaned_query = preprocess_text(query)

        if not cleaned_query:
            raise ValueError("Kata kunci tidak memiliki teks yang dapat dicari.")

        query_vector = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()

        results = self.df.copy()
        results["similarity_score"] = scores

        results = results[results["similarity_score"] > 0].copy()
        results = results.sort_values(
            by="similarity_score",
            ascending=False,
            kind="stable",
        )

        columns = [
            "review_id",
            "review_text",
            "similarity_score",
        ]

        if "product" in results.columns:
            columns.append("product")

        if "price" in results.columns:
            columns.append("price")

        if "packaging" in results.columns:
            columns.append("packaging")

        if "aroma" in results.columns:
            columns.append("aroma")

        return results[columns].head(top_k).reset_index(drop=True)
