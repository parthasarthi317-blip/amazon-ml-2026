import csv
import random
import sqlite3
import time
from pathlib import Path

from normalize import normalize_name


# =========================================================
# PATHS
# =========================================================

ROOT = Path(
    "6ab10eb3b23ba_student_resource/student_resource"
)

TRAIN = ROOT / "dataset/train"

DB_PATH = Path("models/blocking_index.db")
GT_FILE = TRAIN / "train_ground_truth.tsv"
S1_FILE = TRAIN / "train_source1.tsv"


# =========================================================
# SETTINGS
# =========================================================

SAMPLE_SIZE = 2000
SEED = 42

# Retrieve once up to the largest K, then evaluate all K.
MAX_K = 1000

K_VALUES = [50, 100, 250, 500, 1000]


# =========================================================
# GROUND TRUTH SAMPLE
# =========================================================

def sample_ground_truth(path, k, seed=42):

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

            if row["entity_id"] in required_ids:
                records[row["entity_id"]] = row

    return records


# =========================================================
# FTS QUERY
# =========================================================

def make_trigram_query(text):

    if not text:
        return ""

    text = text.strip()

    if len(text) < 3:
        return ""

    grams = sorted({
        text[i:i + 3]
        for i in range(len(text) - 2)
    })

    if not grams:
        return ""

    return " OR ".join(
        '"' + gram.replace('"', '""') + '"'
        for gram in grams
    )


# =========================================================
# MAIN
# =========================================================

print("=" * 90)
print("FULL TRIGRAM RETRIEVAL BENCHMARK")
print("=" * 90)

print("\nSampling ground truth...")

sample = sample_ground_truth(
    GT_FILE,
    SAMPLE_SIZE,
    SEED
)

true_by_s1 = {}

total_true_pairs = 0

for row in sample:

    s1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"].strip()

    if value:

        matched = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    else:

        matched = []

    true_by_s1[s1_id] = matched
    total_true_pairs += len(matched)


print(
    f"S1 entities: {len(true_by_s1):,}"
)

print(
    f"True pairs:  {total_true_pairs:,}"
)


# =========================================================
# Load S1 records
# =========================================================

print("\nLoading S1 records...")

s1_records = load_s1_records(
    S1_FILE,
    set(true_by_s1.keys())
)

print(
    f"Loaded S1: {len(s1_records):,}"
)


# =========================================================
# Open database
# =========================================================

print("\nOpening database...")

conn = sqlite3.connect(DB_PATH)

conn.execute(
    "PRAGMA temp_store=MEMORY"
)


# =========================================================
# Statistics containers
# =========================================================

# For every K:
pair_hits = {
    k: 0
    for k in K_VALUES
}

full_entity_hits = {
    k: 0
    for k in K_VALUES
}

entities_with_matches = sum(
    1
    for ids in true_by_s1.values()
    if ids
)

candidate_counts = []
query_times = []


# =========================================================
# Retrieval
# =========================================================

print("\nStarting retrieval...\n")

processed = 0


for s1_id, row in s1_records.items():

    name = normalize_name(
        row["business_name"]
    )

    country = row["country"].casefold().strip()

    # -----------------------------------------------------
    # Build a query from BOTH name representations.
    # -----------------------------------------------------

    queries = []

    if name["core"]:

        q = make_trigram_query(
            name["core"]
        )

        if q:
            queries.append(q)

    if name["translit"]:

        q = make_trigram_query(
            name["translit"]
        )

        if q:
            queries.append(q)

    if not queries:

        ranked_candidates = []

    else:

        # Combine both representations into one OR query.
        full_query = " OR ".join(queries)

        start = time.perf_counter()

        results = conn.execute(
            """
            SELECT
                f.rowid,
                r.entity_id,
                bm25(name_fts_full) AS score
            FROM name_fts_full AS f
            JOIN records AS r
                ON r.rowid = f.rowid
            WHERE name_fts_full MATCH ?
              AND r.country = ?
            ORDER BY bm25(name_fts_full)
            LIMIT ?
            """,
            (
                full_query,
                country,
                MAX_K,
            )
        ).fetchall()

        elapsed = (
            time.perf_counter()
            - start
        )

        query_times.append(
            elapsed
        )

        ranked_candidates = [
            result[1]
            for result in results
        ]

    candidate_counts.append(
        len(ranked_candidates)
    )

    # -----------------------------------------------------
    # Evaluate every K
    # -----------------------------------------------------

    true_ids = set(
        true_by_s1[s1_id]
    )

    for k in K_VALUES:

        candidates = set(
            ranked_candidates[:k]
        )

        hit_count = len(
            true_ids.intersection(
                candidates
            )
        )

        pair_hits[k] += hit_count

        if true_ids:

            if true_ids.issubset(
                candidates
            ):

                full_entity_hits[k] += 1

    processed += 1

    if processed % 100 == 0:

        print(
            f"Processed "
            f"{processed:,}/"
            f"{len(s1_records):,}"
        )


conn.close()


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 90)
print("RETRIEVAL RESULTS")
print("=" * 90)

print(
    f"S1 entities evaluated : "
    f"{len(s1_records):,}"
)

print(
    f"True matched pairs     : "
    f"{total_true_pairs:,}"
)

print()

print(
    f"{'K':>8} "
    f"{'Pair Recall':>15} "
    f"{'Full Entity Recall':>20}"
)

print("-" * 48)

for k in K_VALUES:

    pair_recall = (
        pair_hits[k]
        / total_true_pairs
        if total_true_pairs
        else 0
    )

    entity_recall = (
        full_entity_hits[k]
        / entities_with_matches
        if entities_with_matches
        else 0
    )

    print(
        f"{k:>8} "
        f"{pair_recall:>14.2%} "
        f"{entity_recall:>19.2%}"
    )


# =========================================================
# Candidate statistics
# =========================================================

print("\n" + "=" * 90)
print("CANDIDATE / SPEED STATISTICS")
print("=" * 90)

if candidate_counts:

    sorted_counts = sorted(
        candidate_counts
    )

    mean_candidates = (
        sum(candidate_counts)
        / len(candidate_counts)
    )

    median_candidates = sorted_counts[
        len(sorted_counts) // 2
    ]

    p95 = sorted_counts[
        int(
            0.95 *
            (len(sorted_counts) - 1)
        )
    ]

    p99 = sorted_counts[
        int(
            0.99 *
            (len(sorted_counts) - 1)
        )
    ]

    print(
        f"Mean returned candidates : "
        f"{mean_candidates:.2f}"
    )

    print(
        f"Median                    : "
        f"{median_candidates}"
    )

    print(
        f"P95                       : "
        f"{p95}"
    )

    print(
        f"P99                       : "
        f"{p99}"
    )


if query_times:

    avg_ms = (
        sum(query_times)
        / len(query_times)
        * 1000
    )

    qps = (
        1000 / avg_ms
        if avg_ms > 0
        else 0
    )

    print(
        f"Average query time        : "
        f"{avg_ms:.2f} ms"
    )

    print(
        f"Queries / second          : "
        f"{qps:.2f}"
    )


print("\nBenchmark complete.")