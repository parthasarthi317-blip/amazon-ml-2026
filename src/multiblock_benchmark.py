import csv
import random
import sqlite3
import statistics
from pathlib import Path


from normalize import normalize_name, normalize_address


ROOT = Path(
    "6ab10eb3b23ba_student_resource/student_resource"
)

TRAIN = ROOT / "dataset/train"

DB_PATH = Path("models/blocking_index.db")

GT_FILE = TRAIN / "train_ground_truth.tsv"
S1_FILE = TRAIN / "train_source1.tsv"


SAMPLE_SIZE = 20000
SEED = 42


BLOCKS = [
    ("core_name", "name_core"),
    ("basic_name", "name_basic"),
    ("token_name", "name_token_sorted"),
    ("translit_name", "name_translit"),
    ("address_token", "address_token_sorted"),
    ("address_canonical", "address_canonical"),
]


def sample_ground_truth(path, k):
    rng = random.Random(SEED)
    sample = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):

            if i < k:
                sample.append(row)

            else:
                j = rng.randint(0, i)

                if j < k:
                    sample[j] = row

    return sample


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


print("=" * 90)
print("MULTI-BLOCK BENCHMARK")
print("=" * 90)

print("\nSampling ground truth...")

gt_sample = sample_ground_truth(
    GT_FILE,
    SAMPLE_SIZE
)

s1_ids = {
    row["source1_entity_id"]
    for row in gt_sample
}

true_by_s1 = {}

total_true_pairs = 0

for row in gt_sample:

    ids = row["matched_entity_ids"].strip()

    matched = []

    if ids:

        matched = [
            x.strip()
            for x in ids.split(",")
            if x.strip()
        ]

    true_by_s1[row["source1_entity_id"]] = matched

    total_true_pairs += len(matched)


print(f"S1 sample: {len(s1_ids):,}")
print(f"True pairs: {total_true_pairs:,}")


print("\nLoading S1...")

s1_records = load_s1_records(
    S1_FILE,
    s1_ids
)

print(
    f"Loaded S1 records: "
    f"{len(s1_records):,}"
)


# ---------------------------------------------------------
# Connect to reusable index
# ---------------------------------------------------------

conn = sqlite3.connect(
    DB_PATH
)

# We will use one prepared query per block.
queries = {}

for block_name, column in BLOCKS:

    queries[block_name] = conn.cursor()


# ---------------------------------------------------------
# Build keys for all S1
# ---------------------------------------------------------

print("\nPreparing S1 blocking keys...")

prepared = {}

for s1_id, row in s1_records.items():

    name = normalize_name(
        row["business_name"]
    )

    address = normalize_address(
        row["business_address"]
    )

    country = row["country"].casefold().strip()

    prepared[s1_id] = {
        "country": country,

        "core_name": name["core"],
        "basic_name": name["basic"],
        "token_name": name["token_sorted"],
        "translit_name": name["translit"],

        "address_token": address["token_sorted"],
        "address_canonical": address["canonical"],
    }


# ---------------------------------------------------------
# Candidate generation for every block
# ---------------------------------------------------------

print("\nGenerating candidates...")

candidates_by_block = {
    block_name: {}
    for block_name, _ in BLOCKS
}

for index, (block_name, column) in enumerate(BLOCKS, 1):

    print(
        f"[{index}/{len(BLOCKS)}] "
        f"{block_name}"
    )

    cursor = queries[block_name]

    for s1_id, keys in prepared.items():

        value = keys[block_name]

        if not value:
            candidates_by_block[block_name][s1_id] = set()
            continue

        cursor.execute(
            f"""
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND {column} = ?
            """,
            (
                keys["country"],
                value,
            )
        )

        result = {
            row[0]
            for row in cursor.fetchall()
        }

        candidates_by_block[
            block_name
        ][s1_id] = result


# ---------------------------------------------------------
# Evaluate individual + cumulative stages
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("BLOCKING RESULTS")
print("=" * 90)


current_union = {
    s1_id: set()
    for s1_id in s1_records
}


for block_name, _ in BLOCKS:

    print(
        f"\n>>> ADDING: {block_name}"
    )

    for s1_id in current_union:

        current_union[s1_id].update(
            candidates_by_block[
                block_name
            ].get(
                s1_id,
                set()
            )
        )

    # -----------------------------------------------------
    # Candidate statistics
    # -----------------------------------------------------

    counts = [
        len(current_union[s1_id])
        for s1_id in current_union
    ]

    counts_sorted = sorted(counts)

    mean_candidates = statistics.mean(
        counts
    )

    median_candidates = statistics.median(
        counts
    )

    p95 = counts_sorted[
        int(0.95 * (len(counts_sorted) - 1))
    ]

    p99 = counts_sorted[
        int(0.99 * (len(counts_sorted) - 1))
    ]

    maximum = counts_sorted[-1]

    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    covered_pairs = 0
    full_entities = 0
    entities_with_matches = 0

    for s1_id, true_ids in true_by_s1.items():

        if not true_ids:
            continue

        entities_with_matches += 1

        candidate_set = current_union.get(
            s1_id,
            set()
        )

        covered = sum(
            1
            for entity_id in true_ids
            if entity_id in candidate_set
        )

        covered_pairs += covered

        if covered == len(true_ids):
            full_entities += 1

    pair_recall = (
        covered_pairs / total_true_pairs
        if total_true_pairs
        else 0
    )

    entity_recall = (
        full_entities / entities_with_matches
        if entities_with_matches
        else 0
    )

    print(
        f"Pair recall       : {pair_recall:.2%}"
    )

    print(
        f"Full entity recall: {entity_recall:.2%}"
    )

    print(
        f"Mean candidates   : {mean_candidates:,.2f}"
    )

    print(
        f"Median candidates : {median_candidates:,.0f}"
    )

    print(
        f"P95 candidates    : {p95:,.0f}"
    )

    print(
        f"P99 candidates    : {p99:,.0f}"
    )

    print(
        f"Maximum candidates: {maximum:,.0f}"
    )


conn.close()

print("\nBenchmark complete.")