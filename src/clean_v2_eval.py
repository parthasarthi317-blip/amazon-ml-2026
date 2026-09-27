import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode


# =========================================================
# PATHS
# =========================================================

ROOT = Path("ml challenge dataset/student_resource")
TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

CLEAN_FILE = Path(
    "experiments/clean_validation_predictions.tsv"
)

MODEL_FILE = Path(
    "models/lightgbm_pair_matcher_v2.txt"
)

OUT_FILE = Path(
    "experiments/clean_v2_predictions.tsv"
)

THRESHOLD_FILE = Path(
    "models/clean_threshold_search_v2.tsv"
)

TOP_K = 1000


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

    out = []

    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")

    return " ".join("".join(out).split())


def normalize_name(text):
    basic = normalize(text)

    tokens = basic.split()
    core = list(tokens)

    while core and core[-1] in LEGAL_SUFFIXES:
        core.pop()

    return {
        "basic": basic,
        "core": " ".join(core),
        "token_sorted": " ".join(sorted(tokens)),
        "tokens": tokens,
    }


def translit_name(text):
    if not text:
        return ""

    return normalize(
        unidecode(str(text))
    )


def normalize_address(text):
    basic = normalize(text)
    tokens = basic.split()

    canonical = [
        ADDRESS_ABBREVIATIONS.get(
            token, token
        )
        for token in tokens
    ]

    return {
        "basic": basic,
        "canonical": " ".join(canonical),
        "token_sorted": " ".join(sorted(canonical)),
        "tokens": canonical,
        "numbers": sorted(
            set(
                re.findall(
                    r"\b\d+[a-z]?\b",
                    basic
                )
            )
        ),
    }


def raw_numeric_chunks(text):
    if not text:
        return []

    return re.findall(
        r"\d+(?:[-/]\d+)+|\d+[a-z]?",
        str(text).casefold()
    )


def compact_numeric_signature(chunks):
    out = []

    for x in chunks:
        digits = re.sub(r"\D", "", x)
        if digits:
            out.append(digits)

    return sorted(set(out))


def primary_numeric_chunk(text):
    chunks = raw_numeric_chunks(text)
    return chunks[0] if chunks else ""


def safe_ratio(a, b):
    if not a or not b:
        return 0.0

    return min(len(a), len(b)) / max(
        len(a), len(b)
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
# LOAD
# =========================================================

print("=" * 90)
print("CLEAN V2 EVALUATION")
print("=" * 90)

print("\nLoading clean candidate pool...")

clean = pd.read_csv(
    CLEAN_FILE,
    sep="\t"
)

clean["source1_entity_id"] = (
    clean["source1_entity_id"].astype(str)
)

clean["target_entity_id"] = (
    clean["target_entity_id"].astype(str)
)

valid_s1_ids = sorted(
    clean["source1_entity_id"].unique()
)

print(
    f"Clean candidate rows: "
    f"{len(clean):,}"
)

print(
    f"Validation S1: "
    f"{len(valid_s1_ids):,}"
)


# =========================================================
# GROUND TRUTH
# =========================================================

print("\nLoading full ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt = gt[
    gt["source1_entity_id"].isin(
        valid_s1_ids
    )
]

true_by_s1 = {}

all_true_ids = set()

for _, row in gt.iterrows():

    value = (
        row["matched_entity_ids"]
        if isinstance(
            row["matched_entity_ids"],
            str
        )
        else ""
    )

    ids = {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }

    true_by_s1[
        row["source1_entity_id"]
    ] = ids

    all_true_ids.update(ids)


print(
    f"True pairs: "
    f"{sum(len(x) for x in true_by_s1.values()):,}"
)


# =========================================================
# LOAD RECORDS
# =========================================================

def load_records(path, ids):

    result = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        import csv

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if row["entity_id"] in ids:
                result[
                    row["entity_id"]
                ] = row

    return result


s1_ids = set(valid_s1_ids)

candidate_target_ids = set(
    clean["target_entity_id"]
)

print("\nLoading S1...")

s1 = load_records(
    S1_FILE,
    s1_ids
)

print(
    f"S1 loaded: {len(s1):,}"
)

print("\nLoading candidate targets...")

s2_ids = {
    x for x in candidate_target_ids
    if x.startswith("S2-")
}

s3_ids = {
    x for x in candidate_target_ids
    if x.startswith("S3-")
}

s2 = load_records(
    S2_FILE,
    s2_ids
)

s3 = load_records(
    S3_FILE,
    s3_ids
)

targets = {
    **s2,
    **s3,
}

print(
    f"S2 loaded: {len(s2):,}"
)

print(
    f"S3 loaded: {len(s3):,}"
)

print(
    f"Targets loaded: {len(targets):,}"
)


# =========================================================
# CACHE
# =========================================================

print("\nPreparing normalized cache...")

s1_cache = {}

for entity_id, row in s1.items():

    name_raw = row["business_name"]
    address_raw = (
        row["business_address"]
        or ""
    )

    s1_cache[entity_id] = {
        "name": normalize_name(
            name_raw
        ),
        "translit":
            translit_name(name_raw),
        "address":
            normalize_address(
                address_raw
            ),
        "raw_address":
            address_raw,
        "country":
            row["country"]
            .casefold()
            .strip(),
        "raw_name":
            name_raw,
    }


target_cache = {}

for entity_id, row in targets.items():

    name_raw = row["business_name"]
    address_raw = (
        row["business_address"]
        or ""
    )

    target_cache[entity_id] = {
        "name": normalize_name(
            name_raw
        ),
        "translit":
            translit_name(name_raw),
        "address":
            normalize_address(
                address_raw
            ),
        "raw_address":
            address_raw,
        "country":
            row["country"]
            .casefold()
            .strip(),
        "raw_name":
            name_raw,
    }


# =========================================================
# TRANSLITERATED TF-IDF
# =========================================================

print("\nBuilding transliterated TF-IDF...")

target_ids_ordered = list(
    targets.keys()
)

target_translit_text = [
    target_cache[x]["translit"]
    for x in target_ids_ordered
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

target_pos_map = {
    entity_id: i
    for i, entity_id
    in enumerate(target_ids_ordered)
}


# =========================================================
# UNIQUE QUERY TRANSLIT MATRIX
# =========================================================

query_translit_text = [
    s1_cache[x]["translit"]
    for x in valid_s1_ids
]

query_translit_matrix = (
    translit_vectorizer.transform(
        query_translit_text
    )
)

query_pos_map = {
    entity_id: i
    for i, entity_id
    in enumerate(valid_s1_ids)
}


# =========================================================
# BUILD V2 FEATURES
# =========================================================

print("\nBuilding V2 features...")

feature_rows = []

for i, row in enumerate(
    clean.itertuples(index=False),
    1
):

    sid = row.source1_entity_id
    tid = row.target_entity_id

    a = s1_cache[sid]
    b = target_cache[tid]

    n1 = a["name"]
    n2 = b["name"]

    a1 = a["address"]
    a2 = b["address"]

    # -----------------------------------------------------
    # Name
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Transliteration
    # -----------------------------------------------------

    t1 = a["translit"]
    t2 = b["translit"]

    translit_ratio = (
        fuzz.ratio(
            t1, t2
        ) / 100.0
    )

    translit_token_sort_ratio = (
        fuzz.token_sort_ratio(
            t1, t2
        ) / 100.0
    )

    translit_token_set_ratio = (
        fuzz.token_set_ratio(
            t1, t2
        ) / 100.0
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
            for c in a["raw_name"]
        )
    )

    target_nonlatin = int(
        any(
            ord(c) > 127
            for c in b["raw_name"]
        )
    )

    both_nonlatin = int(
        s1_nonlatin
        and target_nonlatin
    )

    qi = query_pos_map[sid]
    ti = target_pos_map[tid]

    translit_tfidf = float(
        (
            query_translit_matrix[qi]
            @
            target_translit_matrix[ti].T
        )
        .toarray()[0, 0]
    )

    # -----------------------------------------------------
    # Address
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Numeric structure
    # -----------------------------------------------------

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
            & set(raw_num_2)
        )
    )

    numeric_signature_jaccard = jaccard(
        sig_1,
        sig_2
    )

    primary_number_equal_raw = int(
        bool(primary_1)
        and bool(primary_2)
        and
        primary_1 == primary_2
    )

    primary_number_conflict = int(
        bool(primary_1)
        and bool(primary_2)
        and
        primary_1 != primary_2
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
            len(a1["numbers"]),
            len(a2["numbers"])
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

    # -----------------------------------------------------
    # Other
    # -----------------------------------------------------

    country_equal = int(
        a["country"]
        ==
        b["country"]
    )

    is_s2 = int(
        tid.startswith("S2-")
    )

    is_s3 = int(
        tid.startswith("S3-")
    )

    # -----------------------------------------------------
    # Retrieval features from clean candidate pool
    # -----------------------------------------------------

    name_tfidf = float(
        row.name_tfidf
    )

    address_tfidf = float(
        row.address_tfidf
    )

    hybrid_tfidf = float(
        row.hybrid_tfidf
    )

    # -----------------------------------------------------
    # Interactions
    # -----------------------------------------------------

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

    feature_rows.append({

        "source1_entity_id": sid,
        "target_entity_id": tid,

        "name_tfidf": name_tfidf,
        "address_tfidf": address_tfidf,
        "hybrid_tfidf": hybrid_tfidf,

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
            both_nonlatin,

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

        "country_equal":
            country_equal,

        "is_s2":
            is_s2,

        "is_s3":
            is_s3,

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
            f"Features: "
            f"{i:,}/{len(clean):,}"
        )


# =========================================================
# DATAFRAME
# =========================================================

features = pd.DataFrame(
    feature_rows
)

print(
    f"\nFeature rows: "
    f"{len(features):,}"
)


# =========================================================
# LOAD V2 MODEL
# =========================================================

print("\nLoading V2 LightGBM...")

model = lgb.Booster(
    model_file=str(
        MODEL_FILE
    )
)

feature_names = model.feature_name()

X = features[
    feature_names
].astype(np.float32)


# =========================================================
# PREDICT
# =========================================================

print("\nPredicting...")

features["probability"] = model.predict(
    X
)
print("\nSaving predictions before threshold search...")

features.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)

print(
    f"Intermediate predictions saved: "
    f"{OUT_FILE}"
)


# =========================================================
# CANDIDATE RECALL
# =========================================================

print("\n" + "=" * 90)
print("CLEAN CANDIDATE RECALL")
print("=" * 90)

pair_hits = 0
true_pairs = 0
entity_hits = 0
matched_entities = 0

for sid in valid_s1_ids:

    truth = true_by_s1.get(
        sid,
        set()
    )

    retrieved = set(
        clean.loc[
            clean[
                "source1_entity_id"
            ] == sid,
            "target_entity_id"
        ]
    )

    true_pairs += len(truth)

    pair_hits += len(
        truth & retrieved
    )

    if truth:
        matched_entities += 1

        if truth.issubset(
            retrieved
        ):
            entity_hits += 1


candidate_pair_recall = (
    pair_hits / true_pairs
    if true_pairs
    else 0.0
)

candidate_entity_recall = (
    entity_hits / matched_entities
    if matched_entities
    else 0.0
)

print(
    f"Pair recall @1000: "
    f"{candidate_pair_recall:.2%}"
)

print(
    f"Full entity recall @1000: "
    f"{candidate_entity_recall:.2%}"
)


# =========================================================
# FAST F0.5 EVALUATION
# =========================================================

print("\nPreparing fast threshold evaluation...")


def f05_from_counts(
    tp,
    fp,
    fn
):
    """
    F0.5 from TP / FP / FN.
    """

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

    beta = 0.5

    denominator = (
        beta * beta * precision
        + recall
    )

    if denominator == 0:
        score = 0.0
    else:
        score = (
            (1 + beta * beta)
            * precision
            * recall
            / denominator
        )

    return (
        score,
        precision,
        recall
    )


# ---------------------------------------------------------
# IMPORTANT:
# Prepare each S1 only once.
#
# For each S1:
#   probabilities are sorted descending
#   labels are sorted in exactly the same order
#
# Then for any threshold we can find how many predictions
# exist with np.searchsorted.
# ---------------------------------------------------------

print(
    "Precomputing grouped probability arrays..."
)

group_cache = []

for sid, group in features.groupby(
    "source1_entity_id",
    sort=False
):

    probabilities = (
        group["probability"]
        .to_numpy(
            dtype=np.float64
        )
    )

    labels = (
        group["label"]
        .to_numpy(
            dtype=np.int8
        )
    )

    # Sort highest probability first.
    order = np.argsort(
        -probabilities
    )

    probabilities = probabilities[
        order
    ]

    labels = labels[
        order
    ]

    cumulative_tp = np.cumsum(
        labels,
        dtype=np.int32
    )

    true_count = int(
        labels.sum()
    )

    group_cache.append(
        (
            probabilities,
            cumulative_tp,
            true_count
        )
    )


print(
    f"Prepared "
    f"{len(group_cache):,} S1 groups."
)


# =========================================================
# FAST THRESHOLD EVALUATION
# =========================================================

def evaluate_fast(
    threshold
):

    macro_scores = []
    macro_precision = []
    macro_recall = []

    predicted_total = 0
    true_total = 0

    for (
        probabilities,
        cumulative_tp,
        true_count
    ) in group_cache:

        # Number of rows with probability >= threshold.
        predicted_count = (
            probabilities.size
            -
            np.searchsorted(
                probabilities[::-1],
                threshold,
                side="left"
            )
        )

        if predicted_count <= 0:

            tp = 0

            fp = 0

        else:

            tp = int(
                cumulative_tp[
                    predicted_count - 1
                ]
            )

            fp = (
                predicted_count
                - tp
            )

        fn = (
            true_count
            - tp
        )

        score, precision, recall = (
            f05_from_counts(
                tp,
                fp,
                fn
            )
        )

        macro_scores.append(
            score
        )

        macro_precision.append(
            precision
        )

        macro_recall.append(
            recall
        )

        predicted_total += (
            predicted_count
        )

        true_total += (
            true_count
        )

    return {
        "macro_f05":
            float(
                np.mean(
                    macro_scores
                )
            ),

        "macro_precision":
            float(
                np.mean(
                    macro_precision
                )
            ),

        "macro_recall":
            float(
                np.mean(
                    macro_recall
                )
            ),

        "predicted_matches":
            predicted_total,

        "true_matches":
            true_total,
    }


# =========================================================
# THRESHOLD SWEEP
# =========================================================

print("\nSearching thresholds...")

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

results = []

for threshold in thresholds:

    result = evaluate_fast(
        float(threshold)
    )

    results.append({
        "threshold":
            float(threshold),

        **result
    })


results_df = (
    pd.DataFrame(results)
    .sort_values(
        "macro_f05",
        ascending=False
    )
)


best = results_df.iloc[0]


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 90)
print("CLEAN V2 RESULTS")
print("=" * 90)

print(
    f"Candidate pair recall : "
    f"{candidate_pair_recall:.2%}"
)

print(
    f"Candidate entity recall: "
    f"{candidate_entity_recall:.2%}"
)

print()

print(
    f"Best threshold        : "
    f"{best['threshold']:.2f}"
)

print(
    f"Best Macro F0.5       : "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Macro precision       : "
    f"{best['macro_precision']:.6f}"
)

print(
    f"Macro recall          : "
    f"{best['macro_recall']:.6f}"
)

print(
    f"Predicted matches     : "
    f"{int(best['predicted_matches']):,}"
)

print(
    f"True matches          : "
    f"{int(best['true_matches']):,}"
)


print("\nTop thresholds:")

print(
    results_df.head(15).to_string(
        index=False
    )
)


# =========================================================
# SAVE
# =========================================================

features.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)

results_df.to_csv(
    THRESHOLD_FILE,
    sep="\t",
    index=False
)


print("\nSaved:")
print(
    f"Predictions: {OUT_FILE}"
)

print(
    f"Thresholds : {THRESHOLD_FILE}"
)

print("\nDone.")