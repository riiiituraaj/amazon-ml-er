import re
import unicodedata


# ---------------------------------------------------------------------------
# BUSINESS NAME REPLACEMENTS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ADDRESS REPLACEMENTS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BASIC UNICODE FUNCTIONS
# ---------------------------------------------------------------------------

def unicode_normalize(text: str) -> str:
    """
    Normalize Unicode characters while preserving non-Latin scripts.
    """

    if text is None:
        return ""

    return unicodedata.normalize("NFKC", str(text))


def basic_clean(text: str) -> str:
    """
    Unicode-safe basic text cleaning.

    Keeps:
    - Unicode letters
    - Unicode combining marks
    - Unicode numbers
    - whitespace

    Converts punctuation and symbols into spaces.
    """

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()

    if text.strip() in {"nan", "none", "null"}:
        return ""

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if char.isspace():
            cleaned.append(" ")

        elif category[0] in {"L", "M", "N"}:
            cleaned.append(char)

        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# ORIGINAL E001 NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_name(text: str) -> str:
    """
    Conservative normalized business name.
    """

    text = basic_clean(text)

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

    return re.sub(r"\s+", " ", text).strip()


def normalize_address(text: str) -> str:
    """
    Conservative normalized address.
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

    return re.sub(r"\s+", " ", text).strip()


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


# ---------------------------------------------------------------------------
# E004 - LATIN DIACRITIC FOLDING
# ---------------------------------------------------------------------------

def fold_latin_diacritics(text: str) -> str:
    """
    Remove accent marks from Latin characters.

    Examples:
        Récord -> Record
        Cáre   -> Care
        Índia  -> India

    Combining marks belonging to non-Latin scripts are preserved.
    """

    if not text:
        return ""

    decomposed = unicodedata.normalize("NFD", str(text))

    result = []

    remove_next_marks = False

    for char in decomposed:
        category = unicodedata.category(char)

        # Combining mark.
        if category.startswith("M"):
            if remove_next_marks:
                continue

            result.append(char)
            continue

        # This is a base character.
        char_name = unicodedata.name(char, "")

        # Only Latin base characters get their accents removed.
        remove_next_marks = char_name.startswith("LATIN ")

        result.append(char)

    # IMPORTANT:
    # NFC recomposes the characters that remain,
    # but no Latin accent marks remain because they
    # were removed above.
    return unicodedata.normalize("NFC", "".join(result))


# ---------------------------------------------------------------------------
# E004 - IMPROVED NAME NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_name_v2(text: str) -> str:
    """
    Improved business-name normalization.

    Pipeline:

        raw text
            ↓
        basic Unicode cleaning
            ↓
        Latin accent removal
            ↓
        legal/business abbreviation normalization
    """

    text = basic_clean(text)

    if not text:
        return ""

    text = fold_latin_diacritics(text)

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

    return re.sub(r"\s+", " ", text).strip()


def normalize_name_token_sorted(text: str) -> str:
    """
    Improved normalized representation that is
    insensitive to word order.
    """

    normalized = normalize_name_v2(text)

    if not normalized:
        return ""

    tokens = normalized.split()

    return " ".join(sorted(tokens))

# ---------------------------------------------------------------------------
# E005 - ADDRESS COMPONENTS
# ---------------------------------------------------------------------------

def extract_address_components(text: str) -> dict[str, list[str]]:
    """
    Extract useful address components for blocking.

    Returns:
        {
            "numbers": [...],
            "postal_codes": [...],
            "tokens": [...]
        }

    This is intentionally heuristic and does not depend on
    external geographic databases.
    """

    normalized = normalize_address(text)

    if not normalized:
        return {
            "numbers": [],
            "postal_codes": [],
            "tokens": [],
        }

    numbers = extract_numbers(normalized)

    # Postal/PIN-style numeric tokens.
    # We use length rather than a fixed country list because
    # the test set contains an unseen country.
    postal_codes = [
        number
        for number in numbers
        if len(number) in {5, 6}
    ]

    tokens = normalized.split()

    return {
        "numbers": numbers,
        "postal_codes": postal_codes,
        "tokens": tokens,
    }