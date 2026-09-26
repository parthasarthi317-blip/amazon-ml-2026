import csv
from pathlib import Path
from collections import Counter

ROOT = Path(
    "6ab10eb3b23ba_student_resource/student_resource/dataset"
)

FILES = {
    "TRAIN S1": ROOT / "train/train_source1.tsv",
    "TRAIN S2": ROOT / "train/train_source2.tsv",
    "TRAIN S3": ROOT / "train/train_source3.tsv",
    "GROUND TRUTH": ROOT / "train/train_ground_truth.tsv",
    "TEST S1": ROOT / "test/test_source1.tsv",
    "TEST S2": ROOT / "test/test_source2.tsv",
    "TEST S3": ROOT / "test/test_source3.tsv",
}


def load_tsv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


print("=" * 80)
print("AMAZON ML CHALLENGE 2026 - DATA AUDIT")
print("=" * 80)

data = {}

for name, path in FILES.items():

    rows = load_tsv(path)
    data[name] = rows

    print(f"\n{name}")
    print("-" * 60)
    print(f"Rows: {len(rows):,}")
    print(f"Columns: {list(rows[0].keys())}")

    # Missing values
    for col in rows[0].keys():
        missing = sum(
            1 for r in rows
            if not r.get(col, "").strip()
        )

        if missing:
            print(f"  {col} missing: {missing:,}")

    # Country distribution
    if "country" in rows[0]:
        countries = Counter(r["country"] for r in rows)

        print("Countries:")
        for country, count in countries.most_common():
            print(f"  {country}: {count:,}")

    print("Sample:")
    for row in rows[:2]:
        print(row)


# ---------------------------------------------------------
# Ground truth analysis
# ---------------------------------------------------------

gt = data["GROUND TRUTH"]

print("\n" + "=" * 80)
print("GROUND TRUTH ANALYSIS")
print("=" * 80)

distribution = Counter()

for row in gt:

    ids = row["matched_entity_ids"].strip()

    if not ids:
        count = 0
    else:
        count = len([
            x for x in ids.split(",")
            if x.strip()
        ])

    distribution[count] += 1


print(f"Ground-truth entities: {len(gt):,}")

print("\nMatches per Source-1 entity:")

for n, count in sorted(distribution.items()):
    print(f"{n} matches : {count:,}")


zero = distribution.get(0, 0)

print("\nSingleton/no-match entities:", f"{zero:,}")
print(
    "Entities with >=1 match:",
    f"{len(gt) - zero:,}"
)

print("\nAudit complete.")