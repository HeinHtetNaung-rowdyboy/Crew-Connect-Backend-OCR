"""Simple offline similarity fallback, no model downloads."""
import re
from difflib import SequenceMatcher


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.casefold()).strip()


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()
