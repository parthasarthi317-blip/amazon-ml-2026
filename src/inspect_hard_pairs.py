import pandas as pd
from pathlib import Path

ROOT = Path("ml challenge dataset/student_resource")
TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"

FP_FILE = Path(
    "experiments/error_analysis/false_positives.tsv"
)

FN_FILE = Path(
    "experiments/error_analysis/false_negatives.tsv"
)


def load_ids(path, ids):
    result = {}

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )

    df = df[
        df["entity_id"].isin(ids)
    ]

    for _, row in df.iterrows():
        result[row["entity_id"]] = row

    return result


# ---------------------------------------------------------
# Load hard cases
# ---------------------------------------------------------

fp = pd.read_csv(
    FP_FILE,
    sep="\t"
)

fn = pd.read_csv(
    FN_FILE,
    sep="\t"
)
fp = fp.nlargest(
    15,
    "probability"
)

fn = fn.nlargest(
    15,
    "probability"
)

all_s1 = set(
    fp["source1_entity_id"]
).union(
    fn["source1_entity_id"]
)

all_targets = set(
    fp["target_entity_id"]
).union(
    fn["target_entity_id"]
)

s1 = load_ids(
    S1_FILE,
    all_s1
)

s2 = load_ids(
    S2_FILE,
    {
        x for x in all_targets
        if x.startswith("S2-")
    }
)

s3 = load_ids(
    S3_FILE,
    {
        x for x in all_targets
        if x.startswith("S3-")
    }
)

targets = {
    **s2,
    **s3
}


def show(title, rows):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for _, r in rows.iterrows():

        sid = r["source1_entity_id"]
        tid = r["target_entity_id"]

        a = s1[sid]
        b = targets[tid]

        print("\n----------------------------------------")

        print(
            f"S1   : {sid}"
        )

        print(
            f"NAME : {a['business_name']}"
        )

        print(
            f"ADDR : {a['business_address']}"
        )

        print(
            f"COUNTRY: {a['country']}"
        )

        print()

        print(
            f"TGT  : {tid}"
        )

        print(
            f"NAME : {b['business_name']}"
        )

        print(
            f"ADDR : {b['business_address']}"
        )

        print(
            f"COUNTRY: {b['country']}"
        )

        print()

        print(
            f"Probability       : "
            f"{r['probability']}"
        )

        print(
            f"Name TF-IDF      : "
            f"{r['name_tfidf']}"
        )

        print(
            f"Address TF-IDF   : "
            f"{r['address_tfidf']}"
        )

        print(
            f"Name ratio       : "
            f"{r['name_ratio']}"
        )

        print(
            f"Address ratio    : "
            f"{r['address_ratio']}"
        )

        print(
            f"Address Jaccard  : "
            f"{r['address_jaccard']}"
        )

        print(
            f"Shared numbers   : "
            f"{r['shared_numbers']}"
        )

        print(
            f"Country equal    : "
            f"{r['country_equal']}"
        )


show(
    "TOP FALSE POSITIVES",
    fp
)

show(
    "HARDEST FALSE NEGATIVES",
    fn
)