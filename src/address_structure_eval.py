import csv
import random
from pathlib import Path

from address_features import build_address_features


ROOT = Path(
    "6ab10eb3b23ba_student_resource/student_resource"
)

TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

SAMPLE_SIZE = 50000
SEED = 42


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
    result = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            if row["entity_id"] in ids:
                result[row["entity_id"]] = row

    return result


print("Sampling ground truth...")
sample = sample_ground_truth(GT_FILE, SAMPLE_SIZE)

s1_ids = set()
s2_ids = set()
s3_ids = set()

for row in sample:

    s1_ids.add(row["source1_entity_id"])

    matched = row["matched_entity_ids"].strip()

    if not matched:
        continue

    for entity_id in matched.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print("Loading relevant records...")

s1 = load_records(S1_FILE, s1_ids)
s2 = load_records(S2_FILE, s2_ids)
s3 = load_records(S3_FILE, s3_ids)


pairs = 0

shared_number = 0
shared_postal = 0
shared_alphanumeric = 0

token_jaccard_sum = 0
canonical_jaccard_sum = 0

token_jaccard_10 = 0
token_jaccard_25 = 0
token_jaccard_50 = 0


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return 0.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


for row in sample:

    source1 = s1.get(row["source1_entity_id"])

    if source1 is None:
        continue

    a = build_address_features(
        source1["business_address"]
    )

    matched = row["matched_entity_ids"].strip()

    if not matched:
        continue

    for entity_id in matched.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            candidate = s2.get(entity_id)
        else:
            candidate = s3.get(entity_id)

        if candidate is None:
            continue

        b = build_address_features(
            candidate["business_address"]
        )

        pairs += 1

        # Numeric overlap
        if set(a["numbers"]) & set(b["numbers"]):
            shared_number += 1

        # Postal overlap
        if set(a["postal_codes"]) & set(b["postal_codes"]):
            shared_postal += 1

        # Alphanumeric identifier overlap
        if (
            set(a["alphanumeric_tokens"])
            & set(b["alphanumeric_tokens"])
        ):
            shared_alphanumeric += 1

        # Token Jaccard
        tj = jaccard(
            a["tokens"],
            b["tokens"]
        )

        cj = jaccard(
            a["canonical_tokens"],
            b["canonical_tokens"]
        )

        token_jaccard_sum += tj
        canonical_jaccard_sum += cj

        if tj >= 0.10:
            token_jaccard_10 += 1

        if tj >= 0.25:
            token_jaccard_25 += 1

        if tj >= 0.50:
            token_jaccard_50 += 1


print("\n" + "=" * 80)
print("ADDRESS STRUCTURE EVALUATION")
print("=" * 80)

print(f"True matched pairs: {pairs:,}")


def percentage(value):
    return value / pairs if pairs else 0


print(
    f"\nShared number: "
    f"{shared_number:,} "
    f"({percentage(shared_number):.2%})"
)

print(
    f"Shared postal code: "
    f"{shared_postal:,} "
    f"({percentage(shared_postal):.2%})"
)

print(
    f"Shared alphanumeric token: "
    f"{shared_alphanumeric:,} "
    f"({percentage(shared_alphanumeric):.2%})"
)

print(
    f"\nAverage token Jaccard: "
    f"{token_jaccard_sum / pairs:.4f}"
)

print(
    f"Average canonical-token Jaccard: "
    f"{canonical_jaccard_sum / pairs:.4f}"
)

print("\nToken Jaccard coverage:")

print(
    f" >= 0.10 : {token_jaccard_10:,} "
    f"({percentage(token_jaccard_10):.2%})"
)

print(
    f" >= 0.25 : {token_jaccard_25:,} "
    f"({percentage(token_jaccard_25):.2%})"
)

print(
    f" >= 0.50 : {token_jaccard_50:,} "
    f"({percentage(token_jaccard_50):.2%})"
)

print("\nDone.")