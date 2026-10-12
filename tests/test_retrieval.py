
"""Unit tests for the SkinFind retrieval engine."""

import unittest

from src.retrieval import SkinFindSearch


class TestSkinFindSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search_engine = SkinFindSearch()

    def test_index_contains_all_reviews(self):
        self.assertEqual(len(self.search_engine.df), 3960)
        self.assertEqual(self.search_engine.tfidf_matrix.shape[0], 3960)

    def test_search_returns_results(self):
        results = self.search_engine.search("serum kulit kering", top_k=5)

        self.assertFalse(results.empty)
        self.assertLessEqual(len(results), 5)

    def test_results_are_sorted_by_similarity(self):
        results = self.search_engine.search("serum kulit kering", top_k=5)
        scores = results["similarity_score"].tolist()

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_similarity_scores_are_positive(self):
        results = self.search_engine.search("serum kulit kering", top_k=5)

        self.assertTrue((results["similarity_score"] > 0).all())

    def test_search_returns_required_columns(self):
        results = self.search_engine.search("pelembap wajah", top_k=3)

        self.assertIn("review_id", results.columns)
        self.assertIn("review_text", results.columns)
        self.assertIn("similarity_score", results.columns)

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.search_engine.search("   ")

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            self.search_engine.search("serum", top_k=0)

    def test_unrelated_query_returns_no_results(self):
        results = self.search_engine.search("xyzqwerty987654")

        self.assertTrue(results.empty)


    def test_positive_product_filter(self):
        results = self.search_engine.search(
            "serum kulit kering",
            top_k=5,
            aspect="product",
            sentiment="positive",
        )

        self.assertFalse(results.empty)
        self.assertTrue((results["product"] == 1).all())

    def test_negative_price_filter(self):
        results = self.search_engine.search(
            "harga produk skincare",
            top_k=5,
            aspect="price",
            sentiment="negative",
        )

        self.assertFalse(results.empty)
        self.assertTrue((results["price"] == -1).all())

    def test_aspect_filter_without_sentiment(self):
        results = self.search_engine.search(
            "serum kulit kering",
            top_k=5,
            aspect="aroma",
        )

        self.assertIn("aroma", results.columns)

    def test_invalid_aspect_is_rejected(self):
        with self.assertRaises(ValueError):
            self.search_engine.search(
                "serum",
                aspect="tekstur",
            )

    def test_sentiment_requires_aspect(self):
        with self.assertRaises(ValueError):
            self.search_engine.search(
                "serum",
                sentiment="positive",
            )

    def test_invalid_sentiment_is_rejected(self):
        with self.assertRaises(ValueError):
            self.search_engine.search(
                "serum",
                aspect="product",
                sentiment="good",
            )

if __name__ == "__main__":
    unittest.main()
