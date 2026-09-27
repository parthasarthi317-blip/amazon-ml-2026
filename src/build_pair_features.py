import csv
import math
import random
import re
from pathlib import Path

from rapidfuzz import fuzz


# =========================================================
# PATHS
# =========================================================

ROOT = Path("ml challenge dataset/student_resource")
TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"

INPUT_FILE = Path("experiments/candidate_pairs_pilot.tsv")

OUTPUT_TRAIN = Path(
    "experiments/pair_features_train.tsv"
)

OUTPUT_VALID = Path(
    "experiments/pair_features_valid.tsv"
)

SEED = 42
TRAIN_FRACTION = 0.80


# =========================================================
# NORMALIZATION
# =========================================================

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

    if not basic:
        return {
            "basic": "",
            "core": "",
            "token_sorted": "",
            "tokens": [],
        }

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


def normalize_address(text):

    basic = normalize(text)

    if not basic:
        return {
            "basic": "",
            "canonical": "",
            "token_sorted": "",
            "tokens": [],
            "numbers": [],
            "postal": [],
        }

    tokens = basic.split()

    canonical_tokens = [
        ADDRESS_ABBREVIATIONS.get(
            token,
            token
        )
        for token in tokens
    ]

    canonical = " ".join(
        canonical_tokens
    )

    return {
        "basic": basic,
        "canonical": canonical,
        "token_sorted": " ".join(
            sorted(canonical_tokens)
        ),
        "tokens": canonical_tokens,

        "numbers": sorted(
            set(
                re.findall(
                    r"\b\d+[a-z]?\b",
                    basic
                )
            )
        ),

        "postal": sorted(
            set(
                re.findall(
                    r"(?<!\d)(?:\d{5}(?:\s?\d{4})?|\d{6})(?!\d)",
                    basic
                )
            )
        ),
    }


# =========================================================
# FILE LOADING
# =========================================================

def load_records(path, required_ids):

    records = {}

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
                records[row["entity_id"]] = row

    return records


# =========================================================
# FEATURE HELPERS
# =========================================================

def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return min(
        len(a),
        len(b)
    ) / max(
        len(a),
        len(b)
    )


def jaccard(a, b):

    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def overlap_count(a, b):

    return len(
        set(a) & set(b)
    )


def first_number(numbers):

    if not numbers:
        return ""

    return numbers[0]


# =========================================================
# LOAD CANDIDATE FILE
# =========================================================

print("=" * 90)
print("PAIRWISE FEATURE BUILDER")
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
# COLLECT REQUIRED IDS
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
# LOAD SOURCE DATA
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


# =========================================================
# CACHE NORMALIZED FIELDS
# =========================================================

print("\nNormalizing records...")

s1_cache = {}
target_cache = {}


for entity_id, row in s1.items():

    s1_cache[entity_id] = {
        "name": normalize_name(
            row["business_name"]
        ),
        "address": normalize_address(
            row["business_address"]
        ),
        "country": row["country"].casefold().strip(),
    }


for entity_id, row in {
    **s2,
    **s3,
}.items():

    target_cache[entity_id] = {
        "name": normalize_name(
            row["business_name"]
        ),
        "address": normalize_address(
            row["business_address"]
        ),
        "country": row["country"].casefold().strip(),
    }


# =========================================================
# BUILD FEATURES
# =========================================================

print("\nBuilding pairwise features...")

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

    source = target_id[:2]

    a = s1_cache.get(s1_id)
    b = target_cache.get(target_id)

    if a is None or b is None:
        continue


    # -----------------------------------------------------
    # Name
    # -----------------------------------------------------

    n1 = a["name"]
    n2 = b["name"]

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
        and n1["token_sorted"]
        == n2["token_sorted"]
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


    # -----------------------------------------------------
    # Address
    # -----------------------------------------------------

    a1 = a["address"]
    a2 = b["address"]

    address_present_1 = int(
        bool(a1["basic"])
    )

    address_present_2 = int(
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
        and a1["canonical"]
        == a2["canonical"]
    )

    address_token_sorted_equal = int(
        bool(a1["token_sorted"])
        and a1["token_sorted"]
        == a2["token_sorted"]
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


    # -----------------------------------------------------
    # Numeric / structural features
    # -----------------------------------------------------

    shared_numbers = overlap_count(
        a1["numbers"],
        a2["numbers"]
    )

    shared_numbers_binary = int(
        shared_numbers > 0
    )

    shared_postal = overlap_count(
        a1["postal"],
        a2["postal"]
    )

    shared_postal_binary = int(
        shared_postal > 0
    )

    first_number_equal = int(
        bool(first_number(a1["numbers"]))
        and
        first_number(a1["numbers"])
        == first_number(a2["numbers"])
    )


    # -----------------------------------------------------
    # Country / source
    # -----------------------------------------------------

    country_equal = int(
        a["country"] == b["country"]
    )

    is_s2 = int(
        source == "S2"
    )

    is_s3 = int(
        source == "S3"
    )


    # -----------------------------------------------------
    # TF-IDF features already produced by retrieval
    # -----------------------------------------------------

    name_tfidf = float(
        candidate["name_tfidf_score"]
    )

    address_tfidf = float(
        candidate["address_tfidf_score"]
    )

    hybrid_tfidf = float(
        candidate["hybrid_tfidf_score"]
    )

    retrieval_rank = 0

    # -----------------------------------------------------
    # Final row
    # -----------------------------------------------------

    feature_rows.append({

        "source1_entity_id": s1_id,
        "target_entity_id": target_id,

        "label": int(
            candidate["label"]
        ),

        # Retrieval features
        # "retrieval_rank": retrieval_rank,
        "name_tfidf": name_tfidf,
        "address_tfidf": address_tfidf,
        "hybrid_tfidf": hybrid_tfidf,

        # Name
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

        # Address
        "address_present_s1":
            address_present_1,

        "address_present_target":
            address_present_2,

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

        # Numbers
        "shared_numbers":
            shared_numbers,

        "shared_numbers_binary":
            shared_numbers_binary,

        "shared_postal":
            shared_postal,

        "shared_postal_binary":
            shared_postal_binary,

        "first_number_equal":
            first_number_equal,

        # Other
        "country_equal":
            country_equal,

        "is_s2":
            is_s2,

        "is_s3":
            is_s3,
    })


    if i % 10_000 == 0:

        print(
            f"Processed "
            f"{i:,}/"
            f"{len(candidate_rows):,}"
        )


# =========================================================
# GROUPED TRAIN / VALID SPLIT
# =========================================================

print("\nCreating S1-level train/validation split...")

unique_s1 = sorted({
    row["source1_entity_id"]
    for row in feature_rows
})

rng = random.Random(SEED)

rng.shuffle(
    unique_s1
)

split_point = int(
    TRAIN_FRACTION
    * len(unique_s1)
)

train_s1 = set(
    unique_s1[:split_point]
)

valid_s1 = set(
    unique_s1[split_point:]
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
# SAVE
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


print("\nWriting training features...")

write_tsv(
    OUTPUT_TRAIN,
    train_rows
)

print(
    f"Train rows: "
    f"{len(train_rows):,}"
)

print("\nWriting validation features...")

write_tsv(
    OUTPUT_VALID,
    valid_rows
)

print(
    f"Validation rows: "
    f"{len(valid_rows):,}"
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
print("FEATURE SUMMARY")
print("=" * 90)

print(
    f"Unique S1 total     : "
    f"{len(unique_s1):,}"
)

print(
    f"Train S1            : "
    f"{len(train_s1):,}"
)

print(
    f"Validation S1       : "
    f"{len(valid_s1):,}"
)

print()

print(
    f"Train pairs         : "
    f"{len(train_rows):,}"
)

print(
    f"Train positives     : "
    f"{train_pos:,}"
)

print(
    f"Validation pairs    : "
    f"{len(valid_rows):,}"
)

print(
    f"Validation positives: "
    f"{valid_pos:,}"
)

print()

print(
    f"Features            : "
    f"{len(fieldnames) - 3}"
)

print("\nSaved:")
print(OUTPUT_TRAIN)
print(OUTPUT_VALID)

print("\nDone.")