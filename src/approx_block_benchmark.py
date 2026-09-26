import csv
import random
import sqlite3
import statistics
from pathlib import Path

from normalize import normalize_name


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
    ("core_prefix", "name_core", "prefix"),
    ("translit_prefix", "name_translit", "prefix"),
    ("core_suffix", "name_core", "suffix"),
    ("translit_suffix", "name_translit", "suffix"),
    ("token_prefix", "name_token_sorted", "prefix"),
]


def sample_ground_truth(path, k):
    rng = random.Random(SEED)
    sample = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if i < k:
                sample.append(row)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = row

    return sample


def load_s1(path, ids):
    records = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            if row["entity_id"] in ids:
                records[row["entity_id"]] = row

    return records


print("=" * 90)
print("FAST APPROXIMATE BLOCKING BENCHMARK")
print("=" * 90)

# ---------------------------------------------------------
# Ground truth sample
# ---------------------------------------------------------

sample = sample_ground_truth(
    GT_FILE,
    SAMPLE_SIZE
)

true_by_s1 = {}
true_pairs = 0

s1_ids = set()

for row in sample:
    s1_id = row["source1_entity_id"]
    s1_ids.add(s1_id)

    value = row["matched_entity_ids"].strip()

    ids = []

    if value:
        ids = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    true_by_s1[s1_id] = ids
    true_pairs += len(ids)


print(f"S1 sample : {len(s1_ids):,}")
print(f"True pairs: {true_pairs:,}")


# ---------------------------------------------------------
# Load S1 sample
# ---------------------------------------------------------

s1_records = load_s1(
    S1_FILE,
    s1_ids
)

# ---------------------------------------------------------
# Prepare blocking keys
# ---------------------------------------------------------

prepared = {}

for s1_id, row in s1_records.items():

    name = normalize_name(
        row["business_name"]
    )

    country = row["country"].casefold().strip()

    prepared[s1_id] = {
        "country": country,
        "core": name["core"],
        "translit": name["translit"],
        "token_sorted": name["token_sorted"],
    }


# ---------------------------------------------------------
# SQLite
# ---------------------------------------------------------

conn = sqlite3.connect(
    DB_PATH
)

conn.execute("PRAGMA temp_store=MEMORY")

# Temporary table containing the query keys.
conn.execute("""
    CREATE TEMP TABLE query_blocks (
        s1_id TEXT,
        block TEXT,
        country TEXT,
        block_value TEXT,
        len_bucket INTEGER
    )
""")

# Index temporary query table.
conn.execute("""
    CREATE INDEX idx_query_blocks
    ON query_blocks(
        block,
        country,
        block_value,
        len_bucket
    )
""")


def make_block_values(value, mode):

    if not value:
        return ""

    if mode == "prefix":
        return value[:3]

    if mode == "suffix":
        return value[-3:]

    raise ValueError(mode)


def expression_for(mode, column):
    if mode == "prefix":
        return f"substr({column}, 1, 3)"

    if mode == "suffix":
        return f"substr({column}, -3)"

    raise ValueError(mode)


# ---------------------------------------------------------
# Run block by block
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("APPROXIMATE BLOCK RESULTS")
print("=" * 90)


cumulative = {
    s1_id: set()
    for s1_id in prepared
}


for block_name, column, mode in BLOCKS:

    print(f"\n>>> {block_name}")

    # Clear query table.
    conn.execute(
        "DELETE FROM query_blocks"
    )

    rows = []

    for s1_id, data in prepared.items():

        value = data[
            "core"
            if column == "name_core"
            else
            "translit"
            if column == "name_translit"
            else
            "token_sorted"
        ]

        if not value:
            continue

        block_value = make_block_values(
            value,
            mode
        )

        bucket = len(value) // 5

        # We search neighbouring length buckets.
        for bucket_offset in (-1, 0, 1):

            rows.append(
                (
                    s1_id,
                    block_name,
                    data["country"],
                    block_value,
                    bucket + bucket_offset
                )
            )

    conn.executemany(
        """
        INSERT INTO query_blocks
        VALUES (?, ?, ?, ?, ?)
        """,
        rows
    )

    conn.commit()

    expression = expression_for(
        mode,
        column
    )

    # -----------------------------------------------------
    # ONE JOIN instead of 20,000 queries
    # -----------------------------------------------------

    sql = f"""
        SELECT
            q.s1_id,
            r.entity_id
        FROM query_blocks q
        JOIN records r
          ON r.country = q.country
         AND {expression} = q.block_value
         AND length(r.{column}) / 5 = q.len_bucket
        WHERE q.block = ?
    """

    matches = conn.execute(
        sql,
        (block_name,)
    ).fetchall()

    for s1_id, entity_id in matches:

        # Don't allow source1 candidates.
        if entity_id.startswith("S2-") or entity_id.startswith("S3-"):
            cumulative[s1_id].add(entity_id)

    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    covered = 0
    full_entities = 0
    entities_with_matches = 0

    for s1_id, true_ids in true_by_s1.items():

        if not true_ids:
            continue

        entities_with_matches += 1

        candidates = cumulative[s1_id]

        hit_count = sum(
            1
            for entity_id in true_ids
            if entity_id in candidates
        )

        covered += hit_count

        if hit_count == len(true_ids):
            full_entities += 1

    pair_recall = (
        covered / true_pairs
        if true_pairs else 0
    )

    entity_recall = (
        full_entities / entities_with_matches
        if entities_with_matches else 0
    )

    # -----------------------------------------------------
    # Candidate statistics
    # -----------------------------------------------------

    counts = [
        len(cumulative[s1_id])
        for s1_id in cumulative
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

    print(f"Pair recall       : {pair_recall:.2%}")
    print(f"Full entity recall: {entity_recall:.2%}")
    print(f"Mean candidates   : {mean_candidates:,.2f}")
    print(f"Median candidates : {median_candidates:,.0f}")
    print(f"P95 candidates    : {p95:,.0f}")
    print(f"P99 candidates    : {p99:,.0f}")
    print(f"Maximum candidates: {maximum:,.0f}")


conn.close()

print("\nBenchmark complete.")