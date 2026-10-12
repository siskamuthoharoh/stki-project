
"""Unit tests for SkinFind text preprocessing."""

import unittest

from src.text_preprocessing import preprocess_text


class TestPreprocessText(unittest.TestCase):

    def test_lowercase_and_punctuation(self):
        self.assertEqual(
            preprocess_text("Produk BAGUS!!!"),
            "produk bagus"
        )

    def test_remove_url(self):
        self.assertEqual(
            preprocess_text("Cek https://contoh.com sekarang"),
            "cek sekarang"
        )

    def test_normalize_whitespace(self):
        self.assertEqual(
            preprocess_text("  Produk   ini   bagus  "),
            "produk ini bagus"
        )
    
    def test_apostrophe_handling(self):
        self.assertEqual(
            preprocess_text("it's worth it"),
            "its worth it"
        )

    def test_preserve_negation(self):
        self.assertEqual(
            preprocess_text("Saya TIDAK suka!!!"),
            "saya tidak suka"
        )

    def test_empty_string(self):
        self.assertEqual(preprocess_text(""), "")

    def test_none_input(self):
        self.assertEqual(preprocess_text(None), "")

    def test_remove_email(self):
        self.assertEqual(
            preprocess_text("Hubungi saya test@example.com ya"),
            "hubungi saya ya"
        )


if __name__ == "__main__":
    unittest.main()
