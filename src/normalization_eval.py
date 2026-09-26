import csv
import random
from pathlib import Path

from rapidfuzz.fuzz import ratio

from normalize import normalize_name, normalize_address


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


def reservoir_sample_ground_truth(path, k, seed=42):
    """
    Randomly sample k ground-truth rows without loading
    the entire 2.2M-row file into memory.
    """
    rng = random.Random(seed)
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


def collect_records(path, required_ids):
    records = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            entity_id = row["entity_id"]

            if entity_id in required_ids:
                records[entity_id] = row

    return records


print("Sampling ground truth...")
sample = reservoir_sample_ground_truth(
    GT_FILE,
    SAMPLE_SIZE,
    SEED
)

print(f"Ground-truth sample: {len(sample):,}")


# ---------------------------------------------------------
# Collect IDs
# ---------------------------------------------------------

s1_ids = set()
s2_ids = set()
s3_ids = set()

for row in sample:

    s1_ids.add(row["source1_entity_id"])

    value = row["matched_entity_ids"].strip()

    if not value:
        continue

    for entity_id in value.split(","):
        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print(f"S1 records needed: {len(s1_ids):,}")
print(f"S2 records needed: {len(s2_ids):,}")
print(f"S3 records needed: {len(s3_ids):,}")


# ---------------------------------------------------------
# Scan source files
# ---------------------------------------------------------

print("\nScanning S1...")
s1_records = collect_records(S1_FILE, s1_ids)

print("Scanning S2...")
s2_records = collect_records(S2_FILE, s2_ids)

print("Scanning S3...")
s3_records = collect_records(S3_FILE, s3_ids)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

pair_count = 0

name_basic = 0
name_core = 0
name_compact = 0
name_translit = 0
name_sorted = 0

name_any = 0

address_basic = 0
address_compact = 0
address_translit = 0
address_sorted = 0

address_any = 0

both_basic = 0
both_core = 0


# ---------------------------------------------------------
# Evaluate true pairs
# ---------------------------------------------------------

for row in sample:

    s1 = s1_records.get(row["source1_entity_id"])

    if s1 is None:
        continue

    s1_name = normalize_name(s1["business_name"])
    s1_addr = normalize_address(s1["business_address"])

    matched = row["matched_entity_ids"].strip()

    if not matched:
        continue

    for entity_id in matched.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            candidate = s2_records.get(entity_id)
        else:
            candidate = s3_records.get(entity_id)

        if candidate is None:
            continue

        pair_count += 1

        c_name = normalize_name(candidate["business_name"])
        c_addr = normalize_address(candidate["business_address"])

        # ------------------------------
        # Name
        # ------------------------------

        nb = (
            s1_name["basic"] != ""
            and s1_name["basic"] == c_name["basic"]
        )

        nc = (
            s1_name["core"] != ""
            and s1_name["core"] == c_name["core"]
        )

        ncomp = (
            s1_name["compact"] != ""
            and s1_name["compact"] == c_name["compact"]
        )

        ntrans = (
            s1_name["translit"] != ""
            and s1_name["translit"] == c_name["translit"]
        )

        nsort = (
            s1_name["token_sorted"] != ""
            and s1_name["token_sorted"] == c_name["token_sorted"]
        )

        if nb:
            name_basic += 1

        if nc:
            name_core += 1

        if ncomp:
            name_compact += 1

        if ntrans:
            name_translit += 1

        if nsort:
            name_sorted += 1

        if nb or nc or ncomp or ntrans or nsort:
            name_any += 1

        # ------------------------------
        # Address
        # ------------------------------

        ab = (
            s1_addr["basic"] != ""
            and s1_addr["basic"] == c_addr["basic"]
        )

        acomp = (
            s1_addr["compact"] != ""
            and s1_addr["compact"] == c_addr["compact"]
        )

        atrans = (
            s1_addr["translit"] != ""
            and s1_addr["translit"] == c_addr["translit"]
        )

        asort = (
            s1_addr["token_sorted"] != ""
            and s1_addr["token_sorted"] == c_addr["token_sorted"]
        )

        if ab:
            address_basic += 1

        if acomp:
            address_compact += 1

        if atrans:
            address_translit += 1

        if asort:
            address_sorted += 1

        if ab or acomp or atrans or asort:
            address_any += 1

        # ------------------------------
        # Combined
        # ------------------------------

        if nb and ab:
            both_basic += 1

        if nc and ab:
            both_core += 1


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("NORMALIZATION EVALUATION")
print("=" * 90)

print(f"True matched pairs evaluated: {pair_count:,}")


def show(label, value):
    pct = value / pair_count if pair_count else 0
    print(f"{label:<30} {value:>10,}   {pct:>8.2%}")


print("\nNAME EXACT-MATCH COVERAGE")
show("Basic", name_basic)
show("Core", name_core)
show("Compact", name_compact)
show("Transliterated", name_translit)
show("Token sorted", name_sorted)
show("ANY representation", name_any)


print("\nADDRESS EXACT-MATCH COVERAGE")
show("Basic", address_basic)
show("Compact", address_compact)
show("Transliterated", address_translit)
show("Token sorted", address_sorted)
show("ANY representation", address_any)


print("\nCOMBINED EXACT SIGNAL")
show("Basic name + address", both_basic)
show("Core name + address", both_core)


print("\nDone.")