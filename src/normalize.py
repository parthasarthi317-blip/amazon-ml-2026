import re
import unicodedata
from unidecode import unidecode


LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "llc",
    "ltd",
    "limited",
    "co",
    "company",
    "plc",
    "pvt",
    "private",
    "llp",
    "lp",
    "pc",
    "sarl",
    "sa",
    "sas",
    "gmbh",
    "ag",
    "bv",
    "srl",
    "spa",
    "pte",
}


# Only relatively safe/general address abbreviations.
# We deliberately DO NOT blindly expand ambiguous tokens such as
# "st" and "ct".
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "rdg": "ridge",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "pky": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "trl": "trail",
    "expy": "expressway",
    "fwy": "freeway",
    "sq": "square",
    "aly": "alley",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "mt": "mount",
    "bl": "building",
}


DIRECTIONAL_ABBREVIATIONS = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}


def basic_normalize(text: str) -> str:
    """
    Conservative normalization.

    Does NOT try to understand the meaning of the text.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()

    # Treat ampersand as a word
    text = text.replace("&", " and ")

    chars = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith(("P", "S")):
            chars.append(" ")
        else:
            chars.append(ch)

    text = "".join(chars)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact_normalize(text: str) -> str:
    return basic_normalize(text).replace(" ", "")


def transliterated_normalize(text: str) -> str:
    text = basic_normalize(text)

    if not text:
        return ""

    text = unidecode(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def core_name_normalize(name: str) -> str:
    text = basic_normalize(name)

    if not text:
        return ""

    tokens = text.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def token_sorted_normalize(text: str) -> str:
    text = basic_normalize(text)

    if not text:
        return ""

    return " ".join(sorted(text.split()))


def canonicalize_address_tokens(address: str) -> list[str]:
    """
    Convert an address into conservative canonical tokens.

    Important:
    - The original/basic address is preserved elsewhere.
    - Unknown abbreviations are NOT guessed.
    - Ambiguous abbreviations are NOT blindly expanded.
    """
    text = basic_normalize(address)

    if not text:
        return []

    tokens = text.split()
    output = []

    for token in tokens:

        # Safe abbreviation expansion
        if token in ADDRESS_ABBREVIATIONS:
            token = ADDRESS_ABBREVIATIONS[token]

        # Directional normalization
        elif token in DIRECTIONAL_ABBREVIATIONS:
            token = DIRECTIONAL_ABBREVIATIONS[token]

        output.append(token)

    return output


def canonical_address(address: str) -> str:
    tokens = canonicalize_address_tokens(address)
    return " ".join(tokens)


def extract_numbers(text: str) -> list[str]:
    """
    Preserve meaningful numeric/alphanumeric address components.

    Examples:
        123
        123a
        12
    """
    if not text:
        return []

    text = basic_normalize(text)

    return re.findall(
        r"\b\d+[a-z]?\b",
        text
    )


def extract_postal_candidates(text: str) -> list[str]:
    """
    Generic postal-code candidates.

    We deliberately do not assume that only US/India exist.
    """
    if not text:
        return []

    text = basic_normalize(text)

    candidates = set()

    # US-style 5 digit / ZIP+4
    candidates.update(
        re.findall(r"(?<!\d)\d{5}(?:\s?\d{4})?(?!\d)", text)
    )

    # 6-digit postal formats such as Indian PINs
    candidates.update(
        re.findall(r"(?<!\d)\d{6}(?!\d)", text)
    )

    return sorted(candidates)


def normalize_name(name: str) -> dict:
    basic = basic_normalize(name)

    return {
        "basic": basic,
        "compact": compact_normalize(name),
        "core": core_name_normalize(name),
        "translit": transliterated_normalize(name),
        "token_sorted": token_sorted_normalize(name),
    }


def normalize_address(address: str) -> dict:
    basic = basic_normalize(address)

    canonical = canonical_address(address)

    return {
        "basic": basic,
        "compact": compact_normalize(address),
        "canonical": canonical,
        "translit": transliterated_normalize(address),
        "token_sorted": token_sorted_normalize(address),
        "numbers": extract_numbers(address),
        "postal": extract_postal_candidates(address),
        "tokens": basic.split() if basic else [],
        "canonical_tokens": canonical.split() if canonical else [],
    }