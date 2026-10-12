
"""Text preprocessing utilities for the SkinFind project."""

import re
import unicodedata


def preprocess_text(text: str) -> str:
    """Normalize a beauty product review for information retrieval."""
    if not isinstance(text, str):
        return ""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Convert text to lowercase.
    text = text.lower()

    # Remove URLs and email addresses.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text,
    )

    # Preserve common English contractions.
    text = re.sub(r"(?<=[a-z])['’](?=[a-z])", "", text)

    # Replace punctuation and symbols with spaces.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove underscores and normalize whitespace.
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text
