import re
from normalize import (
    basic_normalize,
    transliterated_normalize,
    canonicalize_address_tokens,
    extract_numbers,
    extract_postal_candidates,
)


def extract_components(address: str) -> list[str]:
    """
    Preserve comma-separated address components before
    punctuation normalization destroys that structure.
    """
    if not address:
        return []

    parts = []

    for part in str(address).split(","):
        part = basic_normalize(part)

        if part:
            parts.append(part)

    return parts


def extract_alphanumeric_tokens(address: str) -> list[str]:
    """
    Preserve tokens containing both letters and digits,
    e.g. 37a, a12, 16-11 variants after basic normalization.
    """
    text = basic_normalize(address)

    if not text:
        return []

    return re.findall(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b", text)


def build_address_features(address: str) -> dict:
    basic = basic_normalize(address)
    canonical_tokens = canonicalize_address_tokens(address)

    return {
        "basic": basic,
        "translit": transliterated_normalize(address),

        "tokens": basic.split() if basic else [],
        "canonical_tokens": canonical_tokens,

        "numbers": extract_numbers(address),
        "postal_codes": extract_postal_candidates(address),

        "alphanumeric_tokens": extract_alphanumeric_tokens(address),

        "components": extract_components(address),

        "component_count": len(extract_components(address)),

        "token_count": len(basic.split()) if basic else 0,
    }