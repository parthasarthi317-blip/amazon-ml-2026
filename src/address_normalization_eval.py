import csv
import random
from pathlib import Path

from normalize import normalize_address


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

    ids = row["matched_entity_ids"].strip()

    if ids:

        for entity_id in ids.split(","):

            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                s3_ids.add(entity_id)


print("Loading relevant records...")

s1 = load_records(S1_FILE, s1_ids)
s2 = load_records(S2_FILE, s2_ids)
s3 = load_records(S3_FILE, s3_ids)


metrics = {
    "basic": 0,
    "canonical": 0,
    "translit": 0,
    "token_sorted": 0,
    "ANY": 0,
}

pairs = 0


for row in sample:

    source1 = s1.get(row["source1_entity_id"])

    if source1 is None:
        continue

    a1 = normalize_address(source1["business_address"])

    matches = row["matched_entity_ids"].strip()

    if not matches:
        continue

    for entity_id in matches.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            candidate = s2.get(entity_id)
        else:
            candidate = s3.get(entity_id)

        if candidate is None:
            continue

        pairs += 1

        a2 = normalize_address(candidate["business_address"])

        checks = {
            "basic": (
                a1["basic"] != ""
                and a1["basic"] == a2["basic"]
            ),
            "canonical": (
                a1["canonical"] != ""
                and a1["canonical"] == a2["canonical"]
            ),
            "translit": (
                a1["translit"] != ""
                and a1["translit"] == a2["translit"]
            ),
            "token_sorted": (
                a1["token_sorted"] != ""
                and a1["token_sorted"] == a2["token_sorted"]
            ),
        }

        for key, matched in checks.items():

            if matched:
                metrics[key] += 1

        if any(checks.values()):
            metrics["ANY"] += 1


print("\n" + "=" * 80)
print("ADDRESS NORMALIZATION EVALUATION")
print("=" * 80)

print(f"True matched pairs: {pairs:,}")

for key, value in metrics.items():

    percentage = value / pairs if pairs else 0

    print(
        f"{key:<20}"
        f"{value:>10,}"
        f"   {percentage:>8.2%}"
    )

print("\nDone.")