import csv
import random
import statistics
from collections import Counter
from pathlib import Path

from normalize import normalize_name, normalize_address


ROOT = Path(
    "6ab10eb3b23ba_student_resource/student_resource"
)

TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

SAMPLE_SIZE = 20000
SEED = 42


BLOCKS = [
    "name_basic",
    "name_core",
    "name_token_sorted",
    "address_canonical",
    "address_token_sorted",
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


def load_records(path, ids):
    records = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            entity_id = row["entity_id"]

            if entity_id in ids:
                records[entity_id] = row

    return records


def make_keys(row):
    """
    Build multiple conservative blocking keys.
    Country is included because the challenge provides it
    and cross-country business matches should not normally
    be generated.
    """

    country = row["country"].casefold().strip()

    name = normalize_name(row["business_name"])
    address = normalize_address(row["business_address"])

    return {
        "name_basic": (
            f"{country}|{name['basic']}"
            if name["basic"] else ""
        ),

        "name_core": (
            f"{country}|{name['core']}"
            if name["core"] else ""
        ),

        "name_token_sorted": (
            f"{country}|{name['token_sorted']}"
            if name["token_sorted"] else ""
        ),

        "address_canonical": (
            f"{country}|{address['canonical']}"
            if address["canonical"] else ""
        ),

        "address_token_sorted": (
            f"{country}|{address['token_sorted']}"
            if address["token_sorted"] else ""
        ),
    }


# =========================================================
# 1. Sample ground truth
# =========================================================

print("Sampling ground truth...")

sample = sample_ground_truth(
    GT_FILE,
    SAMPLE_SIZE
)

print(f"Sampled S1 entities: {len(sample):,}")


# =========================================================
# 2. Collect required IDs
# =========================================================

s1_ids = set()

s2_ids = set()
s3_ids = set()

true_pairs = []

for row in sample:

    s1_id = row["source1_entity_id"]

    s1_ids.add(s1_id)

    matched = row["matched_entity_ids"].strip()

    if not matched:
        continue

    for candidate_id in matched.split(","):

        candidate_id = candidate_id.strip()

        if not candidate_id:
            continue

        true_pairs.append(
            (s1_id, candidate_id)
        )

        if candidate_id.startswith("S2-"):
            s2_ids.add(candidate_id)

        elif candidate_id.startswith("S3-"):
            s3_ids.add(candidate_id)


print(f"True matched pairs: {len(true_pairs):,}")


# =========================================================
# 3. Load sample S1 + true candidates
# =========================================================

print("\nLoading S1 sample...")

s1_records = load_records(
    S1_FILE,
    s1_ids
)

print("Loading true S2 records...")

s2_true = load_records(
    S2_FILE,
    s2_ids
)

print("Loading true S3 records...")

s3_true = load_records(
    S3_FILE,
    s3_ids
)


# =========================================================
# 4. Build query keys for S1
# =========================================================

query_keys = {
    block: {}
    for block in BLOCKS
}

for s1_id, row in s1_records.items():

    keys = make_keys(row)

    for block in BLOCKS:

        key = keys[block]

        if not key:
            continue

        query_keys[block].setdefault(
            key,
            set()
        ).add(s1_id)


# =========================================================
# 5. Prepare true-pair lookup
# =========================================================

true_pair_set = set(true_pairs)

true_by_s1 = {}

for s1_id, candidate_id in true_pairs:

    true_by_s1.setdefault(
        s1_id,
        []
    ).append(candidate_id)


# =========================================================
# 6. Evaluate true-pair recall
# =========================================================

print("\n" + "=" * 90)
print("TRUE MATCH BLOCKING RECALL")
print("=" * 90)

recall_results = {}

for block in BLOCKS:

    covered_pairs = 0

    entity_full_coverage = 0
    entities_with_matches = 0

    for s1_id, candidate_ids in true_by_s1.items():

        s1 = s1_records.get(s1_id)

        if s1 is None:
            continue

        s1_key = make_keys(s1)[block]

        if not s1_key:
            continue

        matched_candidates = 0

        for candidate_id in candidate_ids:

            if candidate_id.startswith("S2-"):
                candidate = s2_true.get(candidate_id)
            else:
                candidate = s3_true.get(candidate_id)

            if candidate is None:
                continue

            candidate_key = make_keys(candidate)[block]

            if candidate_key == s1_key:

                covered_pairs += 1
                matched_candidates += 1

        if candidate_ids:

            entities_with_matches += 1

            if matched_candidates == len(candidate_ids):

                entity_full_coverage += 1

    pair_recall = (
        covered_pairs / len(true_pairs)
        if true_pairs else 0
    )

    entity_recall = (
        entity_full_coverage / entities_with_matches
        if entities_with_matches else 0
    )

    recall_results[block] = pair_recall

    print(
        f"{block:<25}"
        f"pair recall: {pair_recall:>8.2%}   "
        f"full-entity recall: {entity_recall:>8.2%}"
    )


# =========================================================
# 7. Candidate frequency benchmark
# =========================================================

print("\n" + "=" * 90)
print("SCANNING S2 + S3 FOR CANDIDATE COUNTS")
print("=" * 90)

candidate_frequency = {
    block: Counter()
    for block in BLOCKS
}

query_key_sets = {
    block: set(query_keys[block].keys())
    for block in BLOCKS
}


def scan_source(path):

    print(f"Scanning {path.name}...")

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

            keys = make_keys(row)

            for block in BLOCKS:

                key = keys[block]

                if (
                    key
                    and key in query_key_sets[block]
                ):
                    candidate_frequency[block][key] += 1


scan_source(S2_FILE)
scan_source(S3_FILE)


# =========================================================
# 8. Candidate statistics per S1
# =========================================================

print("\n" + "=" * 90)
print("CANDIDATE COUNT STATISTICS")
print("=" * 90)

for block in BLOCKS:

    counts = []

    for s1_id, row in s1_records.items():

        key = make_keys(row)[block]

        if not key:
            count = 0
        else:
            count = candidate_frequency[block].get(
                key,
                0
            )

        counts.append(count)

    counts_sorted = sorted(counts)

    mean_count = statistics.mean(counts)

    median_count = statistics.median(counts)

    p95_index = int(
        0.95 * (len(counts_sorted) - 1)
    )

    p99_index = int(
        0.99 * (len(counts_sorted) - 1)
    )

    p95 = counts_sorted[p95_index]
    p99 = counts_sorted[p99_index]

    maximum = counts_sorted[-1]

    zero = sum(
        1 for x in counts
        if x == 0
    )

    print(f"\n{block}")

    print(
        f"  Mean candidates   : {mean_count:,.2f}"
    )

    print(
        f"  Median candidates : {median_count:,.0f}"
    )

    print(
        f"  P95 candidates    : {p95:,.0f}"
    )

    print(
        f"  P99 candidates    : {p99:,.0f}"
    )

    print(
        f"  Maximum           : {maximum:,.0f}"
    )

    print(
        f"  Zero candidates   : {zero:,} "
        f"({zero / len(counts):.2%})"
    )


print("\nBenchmark complete.")