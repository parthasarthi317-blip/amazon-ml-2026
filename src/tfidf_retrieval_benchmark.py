import csv
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# CONFIG
# =========================================================

ROOT = Path(
     "ml challenge dataset/student_resource"
)

TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"


# Queries
NUM_QUERIES = 2000

# Distractors
DISTRACTORS_S2 = 100_000
DISTRACTORS_S3 = 100_000

SEED = 42

# Retrieval depths
K_VALUES = [50, 100, 250, 500, 1000]
MAX_K = max(K_VALUES)

# Hybrid score:
#
# final_score =
#     NAME_WEIGHT * name_similarity
#   + ADDRESS_WEIGHT * address_similarity
#
NAME_WEIGHT = 0.70
ADDRESS_WEIGHT = 0.30

# Use country as a hard retrieval filter.
USE_COUNTRY_FILTER = True

# Character TF-IDF
CHAR_NGRAM_RANGE = (3, 3)

# Address word TF-IDF
ADDRESS_NGRAM_RANGE = (1, 2)


# =========================================================
# UTILITY: RESERVOIR SAMPLE GROUND TRUTH
# =========================================================

def sample_ground_truth(path, k, seed=42):
    """
    Randomly sample k rows from a very large TSV file
    without loading the whole file into memory.
    """

    rng = random.Random(seed)

    sample = []

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

        for i, row in enumerate(reader):

            if i < k:

                sample.append(row)

            else:

                j = rng.randint(0, i)

                if j < k:
                    sample[j] = row

    return sample


# =========================================================
# UTILITY: NORMALIZATION
# =========================================================

def basic_normalize(text):
    """
    Lightweight normalization for TF-IDF.
    We intentionally avoid expensive additional
    transformations here because this benchmark is
    specifically testing sparse retrieval.
    """

    if not text:
        return ""

    text = str(text).casefold()

    chars = []

    for ch in text:

        if ch.isalnum() or ch.isspace():

            chars.append(ch)

        else:

            chars.append(" ")

    text = "".join(chars)

    return " ".join(text.split())


def core_name(text):
    """
    Remove trailing legal suffixes conservatively.
    """

    suffixes = {
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

    text = basic_normalize(text)

    if not text:
        return ""

    tokens = text.split()

    while tokens and tokens[-1] in suffixes:

        tokens.pop()

    return " ".join(tokens)


# =========================================================
# LOAD S1
# =========================================================

def load_s1_records(path, required_ids):

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

            entity_id = row["entity_id"]

            if entity_id in required_ids:

                records[entity_id] = row

    return records


# =========================================================
# BUILD QUERY SAMPLE
# =========================================================

print("=" * 90)
print("SPARSE TF-IDF RETRIEVAL BENCHMARK")
print("=" * 90)

print("\nSampling ground truth...")

gt_sample = sample_ground_truth(
    GT_FILE,
    NUM_QUERIES,
    SEED
)

query_ids = {
    row["source1_entity_id"]
    for row in gt_sample
}

true_by_s1 = {}

all_true_target_ids = set()

total_true_pairs = 0

for row in gt_sample:

    s1_id = row["source1_entity_id"]

    matched_value = row["matched_entity_ids"].strip()

    if matched_value:

        matched_ids = [
            x.strip()
            for x in matched_value.split(",")
            if x.strip()
        ]

    else:

        matched_ids = []

    true_by_s1[s1_id] = matched_ids

    all_true_target_ids.update(
        matched_ids
    )

    total_true_pairs += len(matched_ids)


print(
    f"S1 queries              : "
    f"{len(query_ids):,}"
)

print(
    f"Queries with >=1 match  : "
    f"{sum(bool(x) for x in true_by_s1.values()):,}"
)

print(
    f"True matched pairs      : "
    f"{total_true_pairs:,}"
)

print(
    f"Unique true target IDs  : "
    f"{len(all_true_target_ids):,}"
)


# =========================================================
# LOAD S1
# =========================================================

print("\nLoading S1 records...")

s1_records = load_s1_records(
    S1_FILE,
    query_ids
)

print(
    f"S1 records loaded       : "
    f"{len(s1_records):,}"
)


# =========================================================
# SCAN TARGET DATA
# =========================================================

print("\nScanning S2/S3 and selecting distractors...")

rng = random.Random(SEED)

selected_s2 = []
selected_s3 = []

true_s2 = {}
true_s3 = {}

seen_s2 = 0
seen_s3 = 0

start_scan = time.time()


def process_target_file(
    path,
    source_name,
    target_limit,
):

    global selected_s2
    global selected_s3
    global seen_s2
    global seen_s3

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

            # -------------------------------------------------
            # Always retain true positives.
            # -------------------------------------------------

            if entity_id in all_true_target_ids:

                if source_name == "S2":
                    true_s2[entity_id] = row

                else:
                    true_s3[entity_id] = row

                continue

            # -------------------------------------------------
            # Reservoir sample distractors.
            # -------------------------------------------------

            seen += 1

            if len(selected) < target_limit:

                selected.append(row)

            else:

                j = rng.randint(0, seen - 1)

                if j < target_limit:

                    selected[j] = row

    if source_name == "S2":

        seen_s2 = seen

        selected_s2 = selected

    else:

        seen_s3 = seen

        selected_s3 = selected


process_target_file(
    S2_FILE,
    "S2",
    DISTRACTORS_S2
)

print(
    f"S2 distractors selected  : "
    f"{len(selected_s2):,}"
)

process_target_file(
    S3_FILE,
    "S3",
    DISTRACTORS_S3
)

print(
    f"S3 distractors selected  : "
    f"{len(selected_s3):,}"
)

print(
    f"Target scan time         : "
    f"{(time.time() - start_scan) / 60:.2f} min"
)


# =========================================================
# BUILD FINAL TARGET POOL
# =========================================================

target_records = []

target_records.extend(
    true_s2.values()
)

target_records.extend(
    true_s3.values()
)

target_records.extend(
    selected_s2
)

target_records.extend(
    selected_s3
)


# Remove accidental duplicate IDs.
target_by_id = {}

for row in target_records:

    target_by_id[row["entity_id"]] = row


target_records = list(
    target_by_id.values()
)


print("\n" + "=" * 90)
print("TARGET POOL")
print("=" * 90)

print(
    f"Unique true targets       : "
    f"{len(all_true_target_ids):,}"
)

print(
    f"S2 distractors             : "
    f"{len(selected_s2):,}"
)

print(
    f"S3 distractors             : "
    f"{len(selected_s3):,}"
)

print(
    f"Total candidate targets    : "
    f"{len(target_records):,}"
)


# =========================================================
# PREPARE TEXT
# =========================================================

print("\nPreparing text...")

query_name_text = []
query_address_text = []
query_country = []
query_ids_ordered = []

for s1_id in query_ids:

    row = s1_records[s1_id]

    query_ids_ordered.append(
        s1_id
    )

    query_name_text.append(
        core_name(
            row["business_name"]
        )
    )

    query_address_text.append(
        basic_normalize(
            row["business_address"]
        )
    )

    query_country.append(
        row["country"].casefold().strip()
    )


target_name_text = []
target_address_text = []
target_country = []
target_ids = []

for row in target_records:

    target_ids.append(
        row["entity_id"]
    )

    target_name_text.append(
        core_name(
            row["business_name"]
        )
    )

    target_address_text.append(
        basic_normalize(
            row["business_address"]
        )
    )

    target_country.append(
        row["country"].casefold().strip()
    )


# =========================================================
# TF-IDF: NAME
# =========================================================

print("\nFitting NAME character TF-IDF...")

start = time.time()

name_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=CHAR_NGRAM_RANGE,
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
    min_df=1,
)

target_name_matrix = name_vectorizer.fit_transform(
    target_name_text
)

query_name_matrix = name_vectorizer.transform(
    query_name_text
)

print(
    f"Name vocabulary          : "
    f"{len(name_vectorizer.vocabulary_):,}"
)

print(
    f"Name matrix shape        : "
    f"{target_name_matrix.shape}"
)

print(
    f"Name nnz                 : "
    f"{target_name_matrix.nnz:,}"
)

print(
    f"Name TF-IDF time         : "
    f"{(time.time() - start):.2f} sec"
)


# =========================================================
# TF-IDF: ADDRESS
# =========================================================

print("\nFitting ADDRESS word TF-IDF...")

start = time.time()

address_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=ADDRESS_NGRAM_RANGE,
    token_pattern=r"(?u)\b\w+\b",
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
    min_df=1,
)

target_address_matrix = address_vectorizer.fit_transform(
    target_address_text
)

query_address_matrix = address_vectorizer.transform(
    query_address_text
)

print(
    f"Address vocabulary      : "
    f"{len(address_vectorizer.vocabulary_):,}"
)

print(
    f"Address matrix shape    : "
    f"{target_address_matrix.shape}"
)

print(
    f"Address nnz             : "
    f"{target_address_matrix.nnz:,}"
)

print(
    f"Address TF-IDF time     : "
    f"{(time.time() - start):.2f} sec"
)


# =========================================================
# RETRIEVAL
# =========================================================

print("\nStarting sparse retrieval...")

TARGET_CHUNK = 20_000
QUERY_BATCH = 16

num_queries = len(query_ids_ordered)
num_targets = len(target_records)


# ---------------------------------------------------------
# Top-K indices for each retrieval method
# ---------------------------------------------------------

name_top_indices = {
    k: [None] * num_queries
    for k in K_VALUES
}

address_top_indices = {
    k: [None] * num_queries
    for k in K_VALUES
}

hybrid_top_indices = {
    k: [None] * num_queries
    for k in K_VALUES
}


def update_topk(
    current_indices,
    current_scores,
    chunk_scores,
    chunk_start,
    max_k,
):
    """
    Merge current top-k with one new target chunk.

    current_indices:
        shape (batch_size, max_k)

    current_scores:
        shape (batch_size, max_k)

    chunk_scores:
        shape (batch_size, chunk_size)
    """

    batch_size = chunk_scores.shape[0]

    # Combine current candidates + new chunk.
    combined_scores = np.concatenate(
        [
            current_scores,
            chunk_scores,
        ],
        axis=1
    )

    chunk_indices = (
        np.arange(
            chunk_scores.shape[1],
            dtype=np.int32
        )
        + chunk_start
    )

    chunk_indices = np.broadcast_to(
        chunk_indices,
        chunk_scores.shape
    )

    combined_indices = np.concatenate(
        [
            current_indices,
            chunk_indices,
        ],
        axis=1
    )

    # Select largest max_k scores.
    if combined_scores.shape[1] > max_k:

        partition_idx = np.argpartition(
            -combined_scores,
            max_k - 1,
            axis=1
        )[:, :max_k]

        row_idx = (
            np.arange(batch_size)[:, None]
        )

        current_scores = combined_scores[
            row_idx,
            partition_idx
        ]

        current_indices = combined_indices[
            row_idx,
            partition_idx
        ]

    else:

        current_scores = combined_scores
        current_indices = combined_indices

    # Sort selected candidates by score.
    order = np.argsort(
        -current_scores,
        axis=1
    )

    row_idx = (
        np.arange(batch_size)[:, None]
    )

    current_scores = current_scores[
        row_idx,
        order
    ]

    current_indices = current_indices[
        row_idx,
        order
    ]

    return (
        current_indices.astype(np.int32),
        current_scores.astype(np.float32),
    )


# ---------------------------------------------------------
# Retrieval loop
# ---------------------------------------------------------

overall_start = time.time()

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

    # Current best candidates.
    name_indices = np.full(
        (batch_size, MAX_K),
        -1,
        dtype=np.int32
    )

    name_scores = np.full(
        (batch_size, MAX_K),
        -np.inf,
        dtype=np.float32
    )

    address_indices = np.full(
        (batch_size, MAX_K),
        -1,
        dtype=np.int32
    )

    address_scores = np.full(
        (batch_size, MAX_K),
        -np.inf,
        dtype=np.float32
    )

    hybrid_indices = np.full(
        (batch_size, MAX_K),
        -1,
        dtype=np.int32
    )

    hybrid_scores = np.full(
        (batch_size, MAX_K),
        -np.inf,
        dtype=np.float32
    )

    query_country_batch = query_country[
        q_start:q_end
    ]

    for t_start in range(
        0,
        num_targets,
        TARGET_CHUNK
    ):

        t_end = min(
            t_start + TARGET_CHUNK,
            num_targets
        )

        # -------------------------------------------------
        # Sparse cosine similarities
        # -------------------------------------------------

        name_scores_chunk = (
            query_name_matrix[q_start:q_end]
            @
            target_name_matrix[t_start:t_end].T
        ).toarray()

        address_scores_chunk = (
            query_address_matrix[q_start:q_end]
            @
            target_address_matrix[t_start:t_end].T
        ).toarray()

        # -------------------------------------------------
        # Country filtering
        # -------------------------------------------------

        if USE_COUNTRY_FILTER:

            target_country_chunk = np.array(
                target_country[t_start:t_end],
                dtype=object
            )

            for i, country in enumerate(
                query_country_batch
            ):

                mismatch = (
                    target_country_chunk
                    != country
                )

                name_scores_chunk[
                    i,
                    mismatch
                ] = -np.inf

                address_scores_chunk[
                    i,
                    mismatch
                ] = -np.inf

        # -------------------------------------------------
        # Hybrid score
        # -------------------------------------------------

        hybrid_scores_chunk = (
            NAME_WEIGHT
            * name_scores_chunk
            +
            ADDRESS_WEIGHT
            * address_scores_chunk
        )

        # -------------------------------------------------
        # Update top-K
        # -------------------------------------------------

        (
            name_indices,
            name_scores
        ) = update_topk(
            name_indices,
            name_scores,
            name_scores_chunk,
            t_start,
            MAX_K,
        )

        (
            address_indices,
            address_scores
        ) = update_topk(
            address_indices,
            address_scores,
            address_scores_chunk,
            t_start,
            MAX_K,
        )

        (
            hybrid_indices,
            hybrid_scores
        ) = update_topk(
            hybrid_indices,
            hybrid_scores,
            hybrid_scores_chunk,
            t_start,
            MAX_K,
        )

    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    for local_i in range(batch_size):

        global_i = q_start + local_i

        name_result = name_indices[
            local_i
        ].tolist()

        address_result = address_indices[
            local_i
        ].tolist()

        hybrid_result = hybrid_indices[
            local_i
        ].tolist()

        for k in K_VALUES:

            name_top_indices[k][global_i] = (
                name_result[:k]
            )

            address_top_indices[k][global_i] = (
                address_result[:k]
            )

            hybrid_top_indices[k][global_i] = (
                hybrid_result[:k]
            )

    processed = q_end

    print(
        f"Processed queries: "
        f"{processed:,}/{num_queries:,}"
    )


retrieval_time = (
    time.time() - overall_start
)


# =========================================================
# EVALUATION
# =========================================================

print("\n" + "=" * 90)
print("RETRIEVAL RESULTS")
print("=" * 90)

print(
    f"Total queries          : "
    f"{num_queries:,}"
)

print(
    f"Total target pool      : "
    f"{num_targets:,}"
)

print(
    f"Retrieval time         : "
    f"{retrieval_time / 60:.2f} min"
)

print()

print(
    f"{'K':>8}"
    f"{'Name Recall':>18}"
    f"{'Address Recall':>20}"
    f"{'Hybrid Recall':>18}"
)

print("-" * 66)


def calculate_pair_recall(
    predictions,
    k
):

    hits = 0

    # predictions is keyed by K
    prediction_rows = predictions[k]

    for i, s1_id in enumerate(
        query_ids_ordered
    ):

        true_ids = set(
            true_by_s1[s1_id]
        )

        if not true_ids:
            continue

        retrieved_ids = {
            target_ids[idx]
            for idx in prediction_rows[i][:k]
            if idx >= 0
        }

        hits += len(
            true_ids.intersection(
                retrieved_ids
            )
        )

    return (
        hits / total_true_pairs
        if total_true_pairs
        else 0.0
    )


for k in K_VALUES:

    name_recall = calculate_pair_recall(
        name_top_indices,
        k
    )

    address_recall = calculate_pair_recall(
        address_top_indices,
        k
    )

    hybrid_recall = calculate_pair_recall(
        hybrid_top_indices,
        k
    )

    print(
        f"{k:>8}"
        f"{name_recall:>17.2%}"
        f"{address_recall:>19.2%}"
        f"{hybrid_recall:>17.2%}"
    )


# =========================================================
# FULL ENTITY RECALL
# =========================================================

print("\n" + "=" * 90)
print("FULL-ENTITY RECALL")
print("=" * 90)


matched_query_count = sum(
    1
    for x in true_by_s1.values()
    if x
)


def calculate_entity_recall(
    predictions,
    k
):

    full_hits = 0

    prediction_rows = predictions[k]

    for i, s1_id in enumerate(
        query_ids_ordered
    ):

        true_ids = set(
            true_by_s1[s1_id]
        )

        if not true_ids:
            continue

        retrieved_ids = {
            target_ids[idx]
            for idx in prediction_rows[i][:k]
            if idx >= 0
        }

        if true_ids.issubset(
            retrieved_ids
        ):
            full_hits += 1

    return (
        full_hits / matched_query_count
        if matched_query_count
        else 0.0
    )

print(
    f"{'K':>8}"
    f"{'Name':>18}"
    f"{'Address':>18}"
    f"{'Hybrid':>18}"
)

print("-" * 62)

for k in K_VALUES:

    name_entity = calculate_entity_recall(
        name_top_indices,
        k
    )

    address_entity = calculate_entity_recall(
        address_top_indices,
        k
    )

    hybrid_entity = calculate_entity_recall(
        hybrid_top_indices,
        k
    )

    print(
        f"{k:>8}"
        f"{name_entity:>17.2%}"
        f"{address_entity:>17.2%}"
        f"{hybrid_entity:>17.2%}"
    )


print("\nBenchmark complete.")