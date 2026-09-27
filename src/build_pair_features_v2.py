import csv
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd

from rapidfuzz import fuzz
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PATHS
# =========================================================

ROOT = Path("ml challenge dataset/student_resource")
TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"

INPUT_FILE = Path(
    "experiments/candidate_pairs_pilot.tsv"
)

OUTPUT_TRAIN = Path(
    "experiments/pair_features_v2_train.tsv"
)

OUTPUT_VALID = Path(
    "experiments/pair_features_v2_valid.tsv"
)


# =========================================================
# SETTINGS
# =========================================================

SEED = 42
TRAIN_FRACTION = 0.80


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
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(text):

    if not text:
        return ""

    text = str(text).casefold()

    chars = []

    for ch in text:

        if ch.isalnum() or ch.isspace():
            chars.append(ch)

        else:
            chars.append(" ")

    return " ".join(
        "".join(chars).split()
    )


def normalize_name(text):

    basic = normalize(text)

    tokens = basic.split()

    core_tokens = list(tokens)

    while (
        core_tokens
        and core_tokens[-1] in LEGAL_SUFFIXES
    ):
        core_tokens.pop()

    return {
        "basic": basic,
        "core": " ".join(core_tokens),
        "token_sorted": " ".join(
            sorted(tokens)
        ),
        "tokens": tokens,
    }


def translit_name(text):

    if not text:
        return ""

    # Transliterate first, then normalize.
    return normalize(
        unidecode(str(text))
    )


def normalize_address(text):

    basic = normalize(text)

    tokens = basic.split()

    canonical_tokens = [
        ADDRESS_ABBREVIATIONS.get(
            token,
            token
        )
        for token in tokens
    ]

    return {
        "basic": basic,

        "canonical":
            " ".join(canonical_tokens),

        "token_sorted":
            " ".join(
                sorted(canonical_tokens)
            ),

        "tokens":
            canonical_tokens,

        "numbers":
            sorted(
                set(
                    re.findall(
                        r"\b\d+[a-z]?\b",
                        basic
                    )
                )
            ),
    }


# =========================================================
# ADDRESS NUMERIC STRUCTURE
# =========================================================

def raw_numeric_chunks(text):

    if not text:
        return []

    text = str(text).casefold()

    # Examples:
    # 7/1
    # 8-2-293/82
    # 23200
    # 624A
    chunks = re.findall(
        r"\d+(?:[-/]\d+)+|\d+[a-z]?",
        text
    )

    return chunks


def compact_numeric_signature(chunks):

    result = []

    for chunk in chunks:

        digits = re.sub(
            r"\D",
            "",
            chunk
        )

        if digits:
            result.append(digits)

    return sorted(
        set(result)
    )


def primary_numeric_chunk(text):

    chunks = raw_numeric_chunks(text)

    if not chunks:
        return ""

    return chunks[0]


# =========================================================
# UTILITY
# =========================================================

def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return (
        min(len(a), len(b))
        /
        max(len(a), len(b))
    )


def jaccard(a, b):

    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# =========================================================
# LOAD RECORDS
# =========================================================

def load_records(
    path,
    required_ids
):

    result = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if row["entity_id"] in required_ids:
                result[
                    row["entity_id"]
                ] = row

    return result


# =========================================================
# READ CANDIDATES
# =========================================================

print("=" * 90)
print("PAIRWISE FEATURE BUILDER V2")
print("=" * 90)

print("\nReading candidate pairs...")

candidate_rows = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:
        candidate_rows.append(row)


print(
    f"Candidate rows: "
    f"{len(candidate_rows):,}"
)


# =========================================================
# REQUIRED IDS
# =========================================================

s1_ids = {
    row["source1_entity_id"]
    for row in candidate_rows
}

target_ids = {
    row["target_entity_id"]
    for row in candidate_rows
}

s2_ids = {
    x
    for x in target_ids
    if x.startswith("S2-")
}

s3_ids = {
    x
    for x in target_ids
    if x.startswith("S3-")
}


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading S1...")

s1 = load_records(
    S1_FILE,
    s1_ids
)

print(
    f"S1 loaded: {len(s1):,}"
)


print("\nLoading S2...")

s2 = load_records(
    S2_FILE,
    s2_ids
)

print(
    f"S2 loaded: {len(s2):,}"
)


print("\nLoading S3...")

s3 = load_records(
    S3_FILE,
    s3_ids
)

print(
    f"S3 loaded: {len(s3):,}"
)


targets = {
    **s2,
    **s3,
}


# =========================================================
# CACHE
# =========================================================

print("\nPreparing normalized cache...")

s1_cache = {}

for entity_id, row in s1.items():

    s1_name = row["business_name"]

    s1_cache[entity_id] = {

        "name":
            normalize_name(
                s1_name
            ),

        "translit":
            translit_name(
                s1_name
            ),

        "address":
            normalize_address(
                row["business_address"]
            ),

        "raw_address":
            row["business_address"] or "",

        "country":
            row["country"]
            .casefold()
            .strip(),
    }


target_cache = {}

for entity_id, row in targets.items():

    target_name = row["business_name"]

    target_cache[entity_id] = {

        "name":
            normalize_name(
                target_name
            ),

        "translit":
            translit_name(
                target_name
            ),

        "address":
            normalize_address(
                row["business_address"]
            ),

        "raw_address":
            row["business_address"] or "",

        "country":
            row["country"]
            .casefold()
            .strip(),
    }


# =========================================================
# TRANSLITERATED NAME TF-IDF
# =========================================================

print(
    "\nBuilding transliterated "
    "name TF-IDF..."
)

ordered_target_ids = list(
    targets.keys()
)

target_translit_text = [
    target_cache[x]["translit"]
    for x in ordered_target_ids
]

translit_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 3),
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
)

target_translit_matrix = (
    translit_vectorizer.fit_transform(
        target_translit_text
    )
)


# =========================================================
# FEATURE GENERATION
# =========================================================

print("\nBuilding V2 features...")

feature_rows = []


for i, candidate in enumerate(
    candidate_rows,
    1
):

    s1_id = candidate[
        "source1_entity_id"
    ]

    target_id = candidate[
        "target_entity_id"
    ]

    a = s1_cache[s1_id]
    b = target_cache[target_id]

    n1 = a["name"]
    n2 = b["name"]

    a1 = a["address"]
    a2 = b["address"]


    # =====================================================
    # ORIGINAL NAME FEATURES
    # =====================================================

    name_basic_equal = int(
        bool(n1["basic"])
        and n1["basic"] == n2["basic"]
    )

    name_core_equal = int(
        bool(n1["core"])
        and n1["core"] == n2["core"]
    )

    name_token_sorted_equal = int(
        bool(n1["token_sorted"])
        and
        n1["token_sorted"]
        ==
        n2["token_sorted"]
    )

    name_ratio = (
        fuzz.ratio(
            n1["core"],
            n2["core"]
        )
        / 100.0
    )

    name_token_sort_ratio = (
        fuzz.token_sort_ratio(
            n1["core"],
            n2["core"]
        )
        / 100.0
    )

    name_token_set_ratio = (
        fuzz.token_set_ratio(
            n1["core"],
            n2["core"]
        )
        / 100.0
    )

    name_partial_ratio = (
        fuzz.partial_ratio(
            n1["core"],
            n2["core"]
        )
        / 100.0
    )

    name_jaccard = jaccard(
        n1["tokens"],
        n2["tokens"]
    )

    name_length_ratio = safe_ratio(
        n1["core"],
        n2["core"]
    )


    # =====================================================
    # TRANSLITERATED NAME FEATURES
    # =====================================================

    t1 = a["translit"]
    t2 = b["translit"]

    translit_ratio = (
        fuzz.ratio(
            t1,
            t2
        )
        / 100.0
    )

    translit_token_sort_ratio = (
        fuzz.token_sort_ratio(
            t1,
            t2
        )
        / 100.0
    )

    translit_token_set_ratio = (
        fuzz.token_set_ratio(
            t1,
            t2
        )
        / 100.0
    )

    translit_jaccard = jaccard(
        t1.split(),
        t2.split()
    )

    translit_equal = int(
        bool(t1)
        and bool(t2)
        and t1 == t2
    )

    s1_nonlatin = int(
        any(
            ord(c) > 127
            for c in str(
                s1[s1_id]["business_name"]
            )
        )
    )

    target_nonlatin = int(
        any(
            ord(c) > 127
            for c in str(
                targets[target_id]["business_name"]
            )
        )
    )

    both_nonlatin = (
        s1_nonlatin
        and target_nonlatin
    )

    translit_query = (
        translit_vectorizer.transform(
            [t1]
        )
    )

    target_pos = ordered_target_ids.index(
        target_id
    )

    translit_tfidf = float(
        (
            translit_query
            @
            target_translit_matrix[
                target_pos
            ].T
        ).toarray()[0, 0]
    )


    # =====================================================
    # ADDRESS FEATURES
    # =====================================================

    address_present_s1 = int(
        bool(a1["basic"])
    )

    address_present_target = int(
        bool(a2["basic"])
    )

    address_missing_either = int(
        not a1["basic"]
        or not a2["basic"]
    )

    address_basic_equal = int(
        bool(a1["basic"])
        and a1["basic"] == a2["basic"]
    )

    address_canonical_equal = int(
        bool(a1["canonical"])
        and
        a1["canonical"]
        ==
        a2["canonical"]
    )

    address_token_sorted_equal = int(
        bool(a1["token_sorted"])
        and
        a1["token_sorted"]
        ==
        a2["token_sorted"]
    )

    address_ratio = (
        fuzz.ratio(
            a1["basic"],
            a2["basic"]
        )
        / 100.0
    )

    address_token_sort_ratio = (
        fuzz.token_sort_ratio(
            a1["basic"],
            a2["basic"]
        )
        / 100.0
    )

    address_token_set_ratio = (
        fuzz.token_set_ratio(
            a1["basic"],
            a2["basic"]
        )
        / 100.0
    )

    address_jaccard = jaccard(
        a1["tokens"],
        a2["tokens"]
    )

    canonical_jaccard = jaccard(
        a1["canonical"].split(),
        a2["canonical"].split()
    )

    address_length_ratio = safe_ratio(
        a1["basic"],
        a2["basic"]
    )


    # =====================================================
    # BETTER NUMERIC STRUCTURE
    # =====================================================

    raw_num_1 = raw_numeric_chunks(
        a["raw_address"]
    )

    raw_num_2 = raw_numeric_chunks(
        b["raw_address"]
    )

    sig_1 = compact_numeric_signature(
        raw_num_1
    )

    sig_2 = compact_numeric_signature(
        raw_num_2
    )

    primary_1 = primary_numeric_chunk(
        a["raw_address"]
    )

    primary_2 = primary_numeric_chunk(
        b["raw_address"]
    )

    numeric_chunk_exact = int(
        bool(raw_num_1)
        and bool(raw_num_2)
        and bool(
            set(raw_num_1)
            &
            set(raw_num_2)
        )
    )

    numeric_signature_jaccard = jaccard(
        sig_1,
        sig_2
    )

    primary_number_equal_raw = int(
        bool(primary_1)
        and bool(primary_2)
        and primary_1 == primary_2
    )

    primary_number_conflict = int(
        bool(primary_1)
        and bool(primary_2)
        and primary_1 != primary_2
    )

    shared_numbers = len(
        set(a1["numbers"])
        &
        set(a2["numbers"])
    )

    shared_number_ratio = (
        shared_numbers
        /
        max(
            1,
            max(
                len(a1["numbers"]),
                len(a2["numbers"])
            )
        )
    )

    number_count_s1 = len(
        a1["numbers"]
    )

    number_count_target = len(
        a2["numbers"]
    )

    number_count_gap = abs(
        number_count_s1
        -
        number_count_target
    )


    # =====================================================
    # OTHER
    # =====================================================

    country_equal = int(
        a["country"]
        ==
        b["country"]
    )

    is_s2 = int(
        target_id.startswith("S2-")
    )

    is_s3 = int(
        target_id.startswith("S3-")
    )


    # =====================================================
    # RETRIEVAL FEATURES
    # =====================================================

    name_tfidf = float(
        candidate[
            "name_tfidf_score"
        ]
    )

    address_tfidf = float(
        candidate[
            "address_tfidf_score"
        ]
    )

    hybrid_tfidf = float(
        candidate[
            "hybrid_tfidf_score"
        ]
    )


    # =====================================================
    # INTERACTIONS
    # =====================================================

    strong_name = int(
        name_ratio >= 0.85
    )

    very_strong_name = int(
        name_ratio >= 0.95
    )

    strong_translit_name = int(
        translit_ratio >= 0.85
    )

    strong_address = int(
        address_ratio >= 0.75
    )

    weak_name = int(
        name_ratio < 0.40
    )

    strong_name_missing_address = int(
        strong_name
        and address_missing_either
    )

    very_strong_name_missing_address = int(
        very_strong_name
        and address_missing_either
    )

    strong_address_weak_name = int(
        strong_address
        and weak_name
    )

    address_only_risk = int(
        weak_name
        and
        address_ratio >= 0.35
        and
        not primary_number_equal_raw
    )

    name_address_ratio_gap = (
        name_ratio
        -
        address_ratio
    )

    tfidf_name_address_gap = (
        name_tfidf
        -
        address_tfidf
    )


    # =====================================================
    # LABEL
    # =====================================================

    label = int(
        candidate["label"]
    )


    # =====================================================
    # STORE
    # =====================================================

    feature_rows.append({

        "source1_entity_id":
            s1_id,

        "target_entity_id":
            target_id,

        "label":
            label,

        # Retrieval
        "name_tfidf":
            name_tfidf,

        "address_tfidf":
            address_tfidf,

        "hybrid_tfidf":
            hybrid_tfidf,

        # Original name
        "name_basic_equal":
            name_basic_equal,

        "name_core_equal":
            name_core_equal,

        "name_token_sorted_equal":
            name_token_sorted_equal,

        "name_ratio":
            name_ratio,

        "name_token_sort_ratio":
            name_token_sort_ratio,

        "name_token_set_ratio":
            name_token_set_ratio,

        "name_partial_ratio":
            name_partial_ratio,

        "name_jaccard":
            name_jaccard,

        "name_length_ratio":
            name_length_ratio,

        # Transliteration
        "translit_tfidf":
            translit_tfidf,

        "translit_equal":
            translit_equal,

        "translit_ratio":
            translit_ratio,

        "translit_token_sort_ratio":
            translit_token_sort_ratio,

        "translit_token_set_ratio":
            translit_token_set_ratio,

        "translit_jaccard":
            translit_jaccard,

        "s1_nonlatin":
            s1_nonlatin,

        "target_nonlatin":
            target_nonlatin,

        "both_nonlatin":
            int(both_nonlatin),

        # Address
        "address_present_s1":
            address_present_s1,

        "address_present_target":
            address_present_target,

        "address_missing_either":
            address_missing_either,

        "address_basic_equal":
            address_basic_equal,

        "address_canonical_equal":
            address_canonical_equal,

        "address_token_sorted_equal":
            address_token_sorted_equal,

        "address_ratio":
            address_ratio,

        "address_token_sort_ratio":
            address_token_sort_ratio,

        "address_token_set_ratio":
            address_token_set_ratio,

        "address_jaccard":
            address_jaccard,

        "canonical_jaccard":
            canonical_jaccard,

        "address_length_ratio":
            address_length_ratio,

        # Numeric
        "numeric_chunk_exact":
            numeric_chunk_exact,

        "numeric_signature_jaccard":
            numeric_signature_jaccard,

        "primary_number_equal_raw":
            primary_number_equal_raw,

        "primary_number_conflict":
            primary_number_conflict,

        "shared_numbers":
            shared_numbers,

        "shared_number_ratio":
            shared_number_ratio,

        "number_count_s1":
            number_count_s1,

        "number_count_target":
            number_count_target,

        "number_count_gap":
            number_count_gap,

        # Other
        "country_equal":
            country_equal,

        "is_s2":
            is_s2,

        "is_s3":
            is_s3,

        # Interactions
        "strong_name":
            strong_name,

        "very_strong_name":
            very_strong_name,

        "strong_translit_name":
            strong_translit_name,

        "strong_address":
            strong_address,

        "weak_name":
            weak_name,

        "strong_name_missing_address":
            strong_name_missing_address,

        "very_strong_name_missing_address":
            very_strong_name_missing_address,

        "strong_address_weak_name":
            strong_address_weak_name,

        "address_only_risk":
            address_only_risk,

        "name_address_ratio_gap":
            name_address_ratio_gap,

        "tfidf_name_address_gap":
            tfidf_name_address_gap,
    })


    if i % 10_000 == 0:

        print(
            f"Processed "
            f"{i:,}/"
            f"{len(candidate_rows):,}"
        )


# =========================================================
# GROUPED SPLIT
# =========================================================

print(
    "\nCreating S1-level split..."
)

unique_s1 = sorted({
    row["source1_entity_id"]
    for row in feature_rows
})

rng = random.Random(
    SEED
)

rng.shuffle(
    unique_s1
)

split = int(
    TRAIN_FRACTION
    * len(unique_s1)
)

train_s1 = set(
    unique_s1[:split]
)

valid_s1 = set(
    unique_s1[split:]
)

train_rows = [
    row
    for row in feature_rows
    if row["source1_entity_id"]
    in train_s1
]

valid_rows = [
    row
    for row in feature_rows
    if row["source1_entity_id"]
    in valid_s1
]


# =========================================================
# WRITE
# =========================================================

fieldnames = list(
    feature_rows[0].keys()
)


def write_tsv(
    path,
    rows
):

    with open(
        path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t"
        )

        writer.writeheader()
        writer.writerows(rows)


write_tsv(
    OUTPUT_TRAIN,
    train_rows
)

write_tsv(
    OUTPUT_VALID,
    valid_rows
)


# =========================================================
# SUMMARY
# =========================================================

train_pos = sum(
    row["label"] == 1
    for row in train_rows
)

valid_pos = sum(
    row["label"] == 1
    for row in valid_rows
)

print("\n" + "=" * 90)
print("V2 FEATURE SUMMARY")
print("=" * 90)

print(
    f"Total S1          : "
    f"{len(unique_s1):,}"
)

print(
    f"Train S1          : "
    f"{len(train_s1):,}"
)

print(
    f"Validation S1     : "
    f"{len(valid_s1):,}"
)

print(
    f"\nTrain rows        : "
    f"{len(train_rows):,}"
)

print(
    f"Train positives   : "
    f"{train_pos:,}"
)

print(
    f"Validation rows   : "
    f"{len(valid_rows):,}"
)

print(
    f"Validation positives: "
    f"{valid_pos:,}"
)

print(
    f"\nFeature count     : "
    f"{len(fieldnames) - 3}"
)

print("\nSaved:")
print(OUTPUT_TRAIN)
print(OUTPUT_VALID)

print("\nDone.")