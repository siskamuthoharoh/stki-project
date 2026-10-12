
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

ASPECTS = ("price", "packaging", "product", "aroma")
SENTIMENTS = {
    "negative": -1,
    "neutral": 0,
    "positive": 1,
}


class SkinFindSearch:
    """Search skincare reviews using TF-IDF and optional filters."""

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

        self.df["cleaned_text"] = (
            self.df["cleaned_text"].fillna("").astype(str)
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["cleaned_text"]
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        aspect: str | None = None,
        sentiment: str | None = None,
    ) -> pd.DataFrame:
        """Search reviews and optionally filter by aspect and sentiment."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Kata kunci pencarian tidak boleh kosong.")

        if top_k < 1:
            raise ValueError("top_k harus minimal 1.")

        if aspect is not None and aspect not in ASPECTS:
            raise ValueError(
                f"Aspek tidak valid. Pilih salah satu: {ASPECTS}"
            )

        if sentiment is not None:
            sentiment = sentiment.lower()

            if sentiment not in SENTIMENTS:
                raise ValueError(
                    "Sentimen tidak valid. Pilih negative, neutral, atau positive."
                )

            if aspect is None:
                raise ValueError(
                    "Pilih aspek terlebih dahulu jika menggunakan filter sentimen."
                )

        cleaned_query = preprocess_text(query)

        if not cleaned_query:
            raise ValueError("Kata kunci tidak memiliki teks yang dapat dicari.")

        query_vector = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()

        results = self.df.copy()
        results["similarity_score"] = scores

        # Apply filters before selecting the top results.
        if aspect is not None:
            results = results[results[aspect].notna()]

            if sentiment is not None:
                expected_label = SENTIMENTS[sentiment]
                results = results[results[aspect] == expected_label]

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

        if aspect is not None:
            columns.append(aspect)
        else:
            columns.extend(
                column for column in ASPECTS if column in results.columns
            )

        return results[list(dict.fromkeys(columns))].head(top_k).reset_index(
            drop=True
        )
