import re
import unicodedata


# Common business/legal abbreviations.
# We keep this deliberately conservative.
NAME_REPLACEMENTS = {
    "corporation": "corp",
    "company": "co",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "llc": "llc",
    "l.l.c": "llc",
    "pvt ltd": "pvtltd",
    "private limited": "pvtltd",
}


ADDRESS_REPLACEMENTS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "apartment": "apt",
    "suite": "ste",
}


def unicode_normalize(text: str) -> str:
    """
    Normalize Unicode characters while preserving non-Latin scripts.
    """
    text = unicodedata.normalize("NFKC", text)
    return text


def basic_clean(text: str) -> str:
    """
    Unicode-safe basic text cleaning.

    Keeps:
    - Unicode letters
    - Unicode combining marks
    - Unicode numbers
    - whitespace

    Converts punctuation/symbols into spaces.
    """

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()

    # Treat common null-like values as empty.
    if text.strip() in {"nan", "none", "null"}:
        return ""

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        # Preserve whitespace.
        if char.isspace():
            cleaned.append(" ")

        # Preserve letters, combining marks and numbers.
        elif category[0] in {"L", "M", "N"}:
            cleaned.append(char)

        # Replace punctuation/symbols with a space.
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_name(text: str) -> str:
    """
    Create a conservative normalized business name.
    """
    text = basic_clean(text)

    # Longest phrases first.
    for old, new in sorted(
        NAME_REPLACEMENTS.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        text = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            text,
        )

    # Remove repeated whitespace again after replacement.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_address(text: str) -> str:
    """
    Create a conservative normalized address.
    """
    text = basic_clean(text)

    for old, new in sorted(
        ADDRESS_REPLACEMENTS.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        text = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            text,
        )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_numbers(text: str) -> list[str]:
    """
    Extract numeric sequences from a string.
    """
    if not text:
        return []

    return re.findall(r"\d+", str(text))


def name_tokens(text: str) -> list[str]:
    """
    Return whitespace-separated normalized name tokens.
    """
    normalized = normalize_name(text)

    if not normalized:
        return []

    return normalized.split()


def address_tokens(text: str) -> list[str]:
    """
    Return whitespace-separated normalized address tokens.
    """
    normalized = normalize_address(text)

    if not normalized:
        return []

    return normalized.split()