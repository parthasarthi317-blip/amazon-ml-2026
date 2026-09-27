import csv
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PATHS
# =========================================================

ROOT = Path("ml challenge dataset/student_resource")
TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"

VALID_FEATURE_FILE = Path(
    "experiments/pair_features_valid.tsv"
)

MODEL_FILE = Path(
    "models/lightgbm_pair_matcher.txt"
)


# =========================================================
# SETTINGS
# =========================================================

SEED = 42

DISTRACTORS_S2 = 100_000
DISTRACTORS_S3 = 100_000

TOP_K = 1000

TARGET_CHUNK = 20_000
QUERY_BATCH = 16

NAME_WEIGHT = 0.70
ADDRESS_WEIGHT = 0.30

TRANSFER_THRESHOLD = 0.58

THRESHOLDS = np.arange(
    0.30,
    0.81,
    0.01
)


# =========================================================
# NORMALIZATION
# =========================================================

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation",
    "llc", "ltd", "limited", "co", "company",
    "plc", "pvt", "private", "llp", "lp",
    "sarl", "sa", "sas", "gmbh", "ag",
    "bv", "srl", "spa", "pte",
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

        "postal":
            sorted(
                set(
                    re.findall(
                        r"(?<!\d)(?:\d{5}(?:\s?\d{4})?|\d{6})(?!\d)",
                        basic
                    )
                )
            ),
    }


# =========================================================
# FILE HELPERS
# =========================================================

def load_records(path, ids):

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

            if row["entity_id"] in ids:
                result[row["entity_id"]] = row

    return result


def sample_distractors(
    path,
    excluded_ids,
    k,
    seed
):

    rng = random.Random(seed)

    selected = []

    seen = 0

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

            entity_id = row["entity_id"]

            if entity_id in excluded_ids:
                continue

            seen += 1

            if len(selected) < k:

                selected.append(row)

            else:

                j = rng.randint(
                    0,
                    seen - 1
                )

                if j < k:
                    selected[j] = row

    return selected


# =========================================================
# GET EXACT VALIDATION S1 SET + GROUND TRUTH
# =========================================================

print("=" * 90)
print("CLEAN END-TO-END VALIDATION")
print("=" * 90)

print(
    "\nReading validation S1 IDs "
    "and ground truth..."
)

valid_df = pd.read_csv(
    VALID_FEATURE_FILE,
    sep="\t"
)

valid_s1_ids = sorted(
    valid_df[
        "source1_entity_id"
    ].unique()
)

true_by_s1 = {}

for s1_id, group in valid_df.groupby(
    "source1_entity_id"
):

    true_by_s1[s1_id] = set(
        group.loc[
            group["label"] == 1,
            "target_entity_id"
        ]
    )


all_true_ids = set()

for ids in true_by_s1.values():
    all_true_ids.update(ids)


print(
    f"Validation S1: "
    f"{len(valid_s1_ids):,}"
)

print(
    f"True target IDs: "
    f"{len(all_true_ids):,}"
)

print(
    f"True pairs: "
    f"{sum(len(x) for x in true_by_s1.values()):,}"
)


# =========================================================
# LOAD S1
# =========================================================

print("\nLoading S1...")

s1 = load_records(
    S1_FILE,
    set(valid_s1_ids)
)

print(
    f"S1 loaded: {len(s1):,}"
)


# =========================================================
# LOAD TRUE TARGETS
# =========================================================

print("\nLoading all TRUE targets...")

true_s2_ids = {
    x
    for x in all_true_ids
    if x.startswith("S2-")
}

true_s3_ids = {
    x
    for x in all_true_ids
    if x.startswith("S3-")
}

true_s2 = load_records(
    S2_FILE,
    true_s2_ids
)

true_s3 = load_records(
    S3_FILE,
    true_s3_ids
)

print(
    f"True S2: {len(true_s2):,}"
)

print(
    f"True S3: {len(true_s3):,}"
)


# =========================================================
# DISTRACTORS
# =========================================================

print("\nSampling distractors...")

start = time.time()

s2_distractors = sample_distractors(
    S2_FILE,
    all_true_ids,
    DISTRACTORS_S2,
    SEED
)

print(
    f"S2 distractors: "
    f"{len(s2_distractors):,}"
)

s3_distractors = sample_distractors(
    S3_FILE,
    all_true_ids,
    DISTRACTORS_S3,
    SEED + 1
)

print(
    f"S3 distractors: "
    f"{len(s3_distractors):,}"
)

print(
    f"Distractor scan time: "
    f"{(time.time() - start) / 60:.2f} min"
)


# =========================================================
# TARGET POOL
# =========================================================

target_by_id = {}

for row in true_s2.values():
    target_by_id[
        row["entity_id"]
    ] = row

for row in true_s3.values():
    target_by_id[
        row["entity_id"]
    ] = row

for row in s2_distractors:
    target_by_id[
        row["entity_id"]
    ] = row

for row in s3_distractors:
    target_by_id[
        row["entity_id"]
    ] = row


targets = list(
    target_by_id.values()
)

target_ids = [
    row["entity_id"]
    for row in targets
]

print("\nTarget pool:")
print(
    f"Total targets: "
    f"{len(targets):,}"
)


# =========================================================
# PREPARE TEXT
# =========================================================

print("\nPreparing normalized text...")

query_names = []
query_addresses = []
query_countries = []

for s1_id in valid_s1_ids:

    row = s1[s1_id]

    n = normalize_name(
        row["business_name"]
    )

    a = normalize_address(
        row["business_address"]
    )

    query_names.append(
        n["core"]
    )

    query_addresses.append(
        a["basic"]
    )

    query_countries.append(
        row["country"].casefold().strip()
    )


target_names = []
target_addresses = []
target_countries = []

for row in targets:

    n = normalize_name(
        row["business_name"]
    )

    a = normalize_address(
        row["business_address"]
    )

    target_names.append(
        n["core"]
    )

    target_addresses.append(
        a["basic"]
    )

    target_countries.append(
        row["country"].casefold().strip()
    )


# =========================================================
# TF-IDF
# =========================================================

print("\nBuilding NAME TF-IDF...")

name_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 3),
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
)

target_name_matrix = (
    name_vectorizer.fit_transform(
        target_names
    )
)

query_name_matrix = (
    name_vectorizer.transform(
        query_names
    )
)


print(
    "Name matrix:",
    target_name_matrix.shape
)


print("\nBuilding ADDRESS TF-IDF...")

address_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b\w+\b",
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
)

target_address_matrix = (
    address_vectorizer.fit_transform(
        target_addresses
    )
)

query_address_matrix = (
    address_vectorizer.transform(
        query_addresses
    )
)


print(
    "Address matrix:",
    target_address_matrix.shape
)


# =========================================================
# RETRIEVE TOP K
# =========================================================

print("\nRunning CLEAN retrieval...")

num_queries = len(valid_s1_ids)
num_targets = len(targets)

retrieved = [
    []
    for _ in range(num_queries)
]

retrieval_start = time.time()


for q_start in range(
    0,
    num_queries,
    QUERY_BATCH
):

    q_end = min(
        q_start + QUERY_BATCH,
        num_queries
    )

    batch_size = q_end - q_start

    best_indices = np.full(
        (batch_size, TOP_K),
        -1,
        dtype=np.int32
    )

    best_scores = np.full(
        (batch_size, TOP_K),
        -np.inf,
        dtype=np.float32
    )


    for t_start in range(
        0,
        num_targets,
        TARGET_CHUNK
    ):

        t_end = min(
            t_start + TARGET_CHUNK,
            num_targets
        )

        name_scores = (
            query_name_matrix[
                q_start:q_end
            ]
            @
            target_name_matrix[
                t_start:t_end
            ].T
        ).toarray()

        address_scores = (
            query_address_matrix[
                q_start:q_end
            ]
            @
            target_address_matrix[
                t_start:t_end
            ].T
        ).toarray()


        hybrid_scores = (
            NAME_WEIGHT * name_scores
            +
            ADDRESS_WEIGHT * address_scores
        )


        # Country filtering
        target_country_chunk = np.array(
            target_countries[
                t_start:t_end
            ],
            dtype=object
        )

        for i in range(batch_size):

            mismatch = (
                target_country_chunk
                != query_countries[
                    q_start + i
                ]
            )

            hybrid_scores[
                i,
                mismatch
            ] = -np.inf


        # Chunk top-K
        local_k = min(
            TOP_K,
            hybrid_scores.shape[1]
        )

        local_idx = np.argpartition(
            -hybrid_scores,
            local_k - 1,
            axis=1
        )[
            :,
            :local_k
        ]

        rows = np.arange(
            batch_size
        )[:, None]

        local_scores = hybrid_scores[
            rows,
            local_idx
        ]

        local_indices = (
            local_idx + t_start
        )


        # Merge current best + chunk
        combined_scores = np.concatenate(
            [
                best_scores,
                local_scores
            ],
            axis=1
        )

        combined_indices = np.concatenate(
            [
                best_indices,
                local_indices
            ],
            axis=1
        )


        top_idx = np.argpartition(
            -combined_scores,
            TOP_K - 1,
            axis=1
        )[
            :,
            :TOP_K
        ]

        best_scores = combined_scores[
            rows,
            top_idx
        ]

        best_indices = combined_indices[
            rows,
            top_idx
        ]


    # Final sort
    order = np.argsort(
        -best_scores,
        axis=1
    )

    rows = np.arange(
        batch_size
    )[:, None]

    best_scores = best_scores[
        rows,
        order
    ]

    best_indices = best_indices[
        rows,
        order
    ]


    for i in range(batch_size):

        retrieved[
            q_start + i
        ] = [
            (
                int(best_indices[i, j]),
                float(best_scores[i, j])
            )
            for j in range(TOP_K)
            if best_indices[i, j] >= 0
            and np.isfinite(
                best_scores[i, j]
            )
        ]


    print(
        f"Processed queries "
        f"{q_end:,}/{num_queries:,}"
    )


print(
    f"\nRetrieval time: "
    f"{(time.time() - retrieval_start) / 60:.2f} min"
)


# =========================================================
# CANDIDATE RECALL
# =========================================================

print("\n" + "=" * 90)
print("CLEAN CANDIDATE RECALL")
print("=" * 90)


total_true = sum(
    len(x)
    for x in true_by_s1.values()
)


pair_hits = 0
full_hits = 0
matched_entities = 0


for qi, s1_id in enumerate(
    valid_s1_ids
):

    true_ids = true_by_s1[
        s1_id
    ]

    if true_ids:
        matched_entities += 1

    retrieved_ids = {
        target_ids[idx]
        for idx, score in retrieved[qi]
    }

    pair_hits += len(
        true_ids & retrieved_ids
    )

    if true_ids.issubset(
        retrieved_ids
    ):
        full_hits += 1


candidate_pair_recall = (
    pair_hits / total_true
    if total_true
    else 0
)

candidate_entity_recall = (
    full_hits / matched_entities
    if matched_entities
    else 0
)


print(
    f"Pair recall @ {TOP_K}: "
    f"{candidate_pair_recall:.2%}"
)

print(
    f"Full entity recall @ {TOP_K}: "
    f"{candidate_entity_recall:.2%}"
)


# =========================================================
# PAIRWISE FEATURE FUNCTIONS
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


# =========================================================
# CACHE NORMALIZED RECORDS
# =========================================================

print("\nCaching normalized records...")

s1_cache = {}
target_cache = {}

for s1_id in valid_s1_ids:

    row = s1[s1_id]

    s1_cache[s1_id] = {
        "name":
            normalize_name(
                row["business_name"]
            ),

        "address":
            normalize_address(
                row["business_address"]
            ),

        "country":
            row["country"]
            .casefold()
            .strip(),
    }


for row in targets:

    entity_id = row["entity_id"]

    target_cache[entity_id] = {
        "name":
            normalize_name(
                row["business_name"]
            ),

        "address":
            normalize_address(
                row["business_address"]
            ),

        "country":
            row["country"]
            .casefold()
            .strip(),
    }


# =========================================================
# BUILD MODEL FEATURES
# =========================================================

print("\nBuilding clean validation features...")

feature_rows = []

for qi, s1_id in enumerate(
    valid_s1_ids
):

    a = s1_cache[s1_id]

    for rank, (
        target_idx,
        hybrid_score
    ) in enumerate(
        retrieved[qi],
        start=1
    ):

        target_id = target_ids[
            target_idx
        ]

        b = target_cache[
            target_id
        ]

        n1 = a["name"]
        n2 = b["name"]

        a1 = a["address"]
        a2 = b["address"]


        # ---------------------------------------------
        # Recompute individual TF-IDF scores
        # ---------------------------------------------

        name_tfidf = float(
            (
                query_name_matrix[qi]
                @
                target_name_matrix[
                    target_idx
                ].T
            ).toarray()[0, 0]
        )

        address_tfidf = float(
            (
                query_address_matrix[qi]
                @
                target_address_matrix[
                    target_idx
                ].T
            ).toarray()[0, 0]
        )


        # ---------------------------------------------
        # Name features
        # ---------------------------------------------

        name_basic_equal = int(
            bool(n1["basic"])
            and n1["basic"]
            == n2["basic"]
        )

        name_core_equal = int(
            bool(n1["core"])
            and n1["core"]
            == n2["core"]
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
            ) / 100.0
        )

        name_token_sort_ratio = (
            fuzz.token_sort_ratio(
                n1["core"],
                n2["core"]
            ) / 100.0
        )

        name_token_set_ratio = (
            fuzz.token_set_ratio(
                n1["core"],
                n2["core"]
            ) / 100.0
        )

        name_partial_ratio = (
            fuzz.partial_ratio(
                n1["core"],
                n2["core"]
            ) / 100.0
        )

        name_jaccard = jaccard(
            n1["tokens"],
            n2["tokens"]
        )

        name_length_ratio = safe_ratio(
            n1["core"],
            n2["core"]
        )


        # ---------------------------------------------
        # Address features
        # ---------------------------------------------

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
            and
            a1["basic"] == a2["basic"]
        )

        address_canonical_equal = int(
            bool(a1["canonical"])
            and
            a1["canonical"]
            == a2["canonical"]
        )

        address_token_sorted_equal = int(
            bool(a1["token_sorted"])
            and
            a1["token_sorted"]
            == a2["token_sorted"]
        )

        address_ratio = (
            fuzz.ratio(
                a1["basic"],
                a2["basic"]
            ) / 100.0
        )

        address_token_sort_ratio = (
            fuzz.token_sort_ratio(
                a1["basic"],
                a2["basic"]
            ) / 100.0
        )

        address_token_set_ratio = (
            fuzz.token_set_ratio(
                a1["basic"],
                a2["basic"]
            ) / 100.0
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


        # ---------------------------------------------
        # Numeric
        # ---------------------------------------------

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
            bool(a1["numbers"])
            and bool(a2["numbers"])
            and
            a1["numbers"][0]
            ==
            a2["numbers"][0]
        )


        # ---------------------------------------------
        # Other
        # ---------------------------------------------

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


        label = int(
            target_id
            in true_by_s1[s1_id]
        )


        feature_rows.append({

            "source1_entity_id":
                s1_id,

            "target_entity_id":
                target_id,

            "label":
                label,

            "name_tfidf":
                name_tfidf,

            "address_tfidf":
                address_tfidf,

            "hybrid_tfidf":
                float(hybrid_score),

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

            "country_equal":
                country_equal,

            "is_s2":
                is_s2,

            "is_s3":
                is_s3,
        })


print(
    f"Feature rows: "
    f"{len(feature_rows):,}"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading trained LightGBM...")

booster = lgb.Booster(
    model_file=str(
        MODEL_FILE
    )
)


# =========================================================
# MODEL FEATURES
# =========================================================

feature_df = pd.DataFrame(
    feature_rows
)

ID_COLUMNS = [
    "source1_entity_id",
    "target_entity_id",
    "label",
]

FEATURE_COLUMNS = [
    x
    for x in feature_df.columns
    if x not in ID_COLUMNS
]

X = feature_df[
    FEATURE_COLUMNS
].astype(np.float32)


# =========================================================
# PREDICT
# =========================================================

print("\nPredicting...")

probabilities = booster.predict(
    X
)

feature_df[
    "probability"
] = probabilities


# =========================================================
# F0.5
# =========================================================

def f05(
    precision,
    recall
):

    beta = 0.5

    denom = (
        beta * beta * precision
        + recall
    )

    if denom == 0:
        return 0.0

    return (
        (1 + beta * beta)
        * precision
        * recall
        / denom
    )


def evaluate_threshold(
    threshold
):

    scores = []

    macro_precision = []
    macro_recall = []

    predicted_total = 0
    true_total = 0

    for s1_id, group in feature_df.groupby(
        "source1_entity_id",
        sort=False
    ):

        true_ids = set(
            group.loc[
                group["label"] == 1,
                "target_entity_id"
            ]
        )

        predicted_ids = set(
            group.loc[
                group["probability"]
                >= threshold,
                "target_entity_id"
            ]
        )

        tp = len(
            true_ids & predicted_ids
        )

        fp = len(
            predicted_ids - true_ids
        )

        fn = len(
            true_ids - predicted_ids
        )

        if tp + fp == 0:

            precision = (
                1.0
                if tp + fn == 0
                else 0.0
            )

        else:

            precision = (
                tp / (tp + fp)
            )

        if tp + fn == 0:

            recall = 1.0

        else:

            recall = (
                tp / (tp + fn)
            )

        scores.append(
            f05(
                precision,
                recall
            )
        )

        macro_precision.append(
            precision
        )

        macro_recall.append(
            recall
        )

        predicted_total += len(
            predicted_ids
        )

        true_total += len(
            true_ids
        )

    return {
        "macro_f05":
            float(np.mean(scores)),

        "macro_precision":
            float(np.mean(
                macro_precision
            )),

        "macro_recall":
            float(np.mean(
                macro_recall
            )),

        "predicted_matches":
            predicted_total,

        "true_matches":
            true_total,
    }


# =========================================================
# THRESHOLD 0.58
# =========================================================

transfer = evaluate_threshold(
    TRANSFER_THRESHOLD
)

print("\n" + "=" * 90)
print("END-TO-END RESULT @ TRANSFERRED THRESHOLD")
print("=" * 90)

print(
    f"Threshold        : "
    f"{TRANSFER_THRESHOLD:.2f}"
)

print(
    f"Macro F0.5       : "
    f"{transfer['macro_f05']:.6f}"
)

print(
    f"Macro precision  : "
    f"{transfer['macro_precision']:.6f}"
)

print(
    f"Macro recall     : "
    f"{transfer['macro_recall']:.6f}"
)

print(
    f"Predicted matches: "
    f"{transfer['predicted_matches']:,}"
)

print(
    f"True matches     : "
    f"{transfer['true_matches']:,}"
)


# =========================================================
# THRESHOLD SWEEP
# =========================================================

threshold_results = []

for threshold in THRESHOLDS:

    result = evaluate_threshold(
        float(threshold)
    )

    threshold_results.append({
        "threshold":
            float(threshold),

        **result
    })


threshold_df = pd.DataFrame(
    threshold_results
).sort_values(
    "macro_f05",
    ascending=False
)


best = threshold_df.iloc[0]


print("\n" + "=" * 90)
print("CLEAN VALIDATION THRESHOLD SWEEP")
print("=" * 90)

print(
    threshold_df.head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("CLEAN VALIDATION SUMMARY")
print("=" * 90)

print(
    f"Candidate pair recall: "
    f"{candidate_pair_recall:.2%}"
)

print(
    f"Candidate entity recall: "
    f"{candidate_entity_recall:.2%}"
)

print(
    f"F0.5 @ 0.58: "
    f"{transfer['macro_f05']:.6f}"
)

print(
    f"Best pilot F0.5: "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Best pilot threshold: "
    f"{best['threshold']:.2f}"
)


# =========================================================
# SAVE
# =========================================================

output = Path(
    "experiments/clean_validation_predictions.tsv"
)

feature_df.to_csv(
    output,
    sep="\t",
    index=False
)

print(
    f"\nSaved: {output}"
)

print("\nDone.")