import csv
from pathlib import Path

from normalize import normalize_name, normalize_address


ROOT = Path(
    "6ab10eb3b23ba_student_resource/student_resource/dataset"
)

FILES = {
    "S1": ROOT / "train/train_source1.tsv",
    "S2": ROOT / "train/train_source2.tsv",
    "S3": ROOT / "train/train_source3.tsv",
}


def audit_file(label, path, max_examples=12):

    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    changed_basic = 0
    changed_core = 0
    changed_address = 0
    total = 0

    examples = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            total += 1

            raw_name = row["business_name"]
            raw_address = row["business_address"]

            names = normalize_name(raw_name)
            addresses = normalize_address(raw_address)

            if names["name_basic"] != raw_name.casefold():
                changed_basic += 1

            if names["name_core"] != names["name_basic"]:
                changed_core += 1

            if addresses["address_basic"] != raw_address.casefold():
                changed_address += 1

            if len(examples) < max_examples:
                examples.append(
                    (
                        raw_name,
                        names["name_basic"],
                        names["name_core"],
                        raw_address,
                        addresses["address_basic"],
                    )
                )

            # We only need a representative sample for display.
            if total >= 100_000:
                break

    print(f"Rows inspected: {total:,}")
    print(
        f"Name basic changed: "
        f"{changed_basic:,} ({changed_basic / total:.2%})"
    )
    print(
        f"Name core differs from basic: "
        f"{changed_core:,} ({changed_core / total:.2%})"
    )
    print(
        f"Address basic changed: "
        f"{changed_address:,} ({changed_address / total:.2%})"
    )

    print("\nExamples:")

    for i, example in enumerate(examples, 1):

        raw_name, basic_name, core_name, raw_address, basic_address = example

        print(f"\n--- Example {i} ---")

        print("RAW NAME     :", raw_name)
        print("BASIC NAME   :", basic_name)
        print("CORE NAME    :", core_name)

        print("RAW ADDRESS  :", raw_address)
        print("BASIC ADDRESS:", basic_address)


for label, path in FILES.items():
    audit_file(label, path)

print("\nNormalization audit complete.")