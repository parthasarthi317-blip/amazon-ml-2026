import re
from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

INPUT = Path(
    "experiments/clean_validation_predictions.tsv"
)

OUTPUT_DIR = Path("experiments/error_analysis")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

THRESHOLD = 0.77


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(
    INPUT,
    sep="\t"
)

print("=" * 90)
print("CLEAN VALIDATION ERROR ANALYSIS")
print("=" * 90)

print(
    f"Rows loaded: {len(df):,}"
)


# =========================================================
# HELPERS
# =========================================================

def has_number(text):
    if not isinstance(text, str):
        return False

    return bool(
        re.search(
            r"\d",
            text
        )
    )


def has_non_ascii(text):
    if not isinstance(text, str):
        return False

    return any(
        ord(c) > 127
        for c in text
    )


def get_script_category(
    name
):
    if not isinstance(name, str):
        return "missing"

    if not name.strip():
        return "missing"

    if has_non_ascii(name):
        return "non_ascii"

    return "latin_ascii"


# =========================================================
# ADD DIAGNOSTIC COLUMNS
# =========================================================

df["predicted"] = (
    df["probability"]
    >= THRESHOLD
)

df["true_positive"] = (
    (df["label"] == 1)
    & df["predicted"]
)

df["false_positive"] = (
    (df["label"] == 0)
    & df["predicted"]
)

df["false_negative"] = (
    (df["label"] == 1)
    & (~df["predicted"])
)

df["missed_by_retrieval"] = (
    df["label"] == 1
    # All rows here are retrieved, so this is false
)

# The clean file only contains retrieved candidates,
# therefore retrieval misses must be analyzed separately
# from the previously measured 97.24% candidate recall.


# =========================================================
# BASIC MODEL ERROR COUNTS
# =========================================================

tp = int(
    df["true_positive"].sum()
)

fp = int(
    df["false_positive"].sum()
)

fn = int(
    df["false_negative"].sum()
)

print("\n" + "=" * 90)
print("MODEL ERRORS @ THRESHOLD", THRESHOLD)
print("=" * 90)

print(
    f"TP: {tp:,}"
)

print(
    f"FP: {fp:,}"
)

print(
    f"FN: {fn:,}"
)


# =========================================================
# ERROR SUBSETS
# =========================================================

false_negatives = df[
    df["false_negative"]
].copy()

false_positives = df[
    df["false_positive"]
].copy()

true_positives = df[
    df["true_positive"]
].copy()


# =========================================================
# ERROR PROFILE
# =========================================================

if len(false_negatives):

    false_negatives["target_name_script"] = (
        false_negatives[
            "target_entity_id"
        ].astype(str)
    )

    print("\nFalse negative score range:")

    print(
        false_negatives[
            "probability"
        ].describe()
    )


print("\nFalse positive score range:")

if len(false_positives):

    print(
        false_positives[
            "probability"
        ].describe()
    )
else:
    print("None")


# =========================================================
# FEATURE COMPARISON
# =========================================================

FEATURES = [
    "name_tfidf",
    "address_tfidf",
    "hybrid_tfidf",

    "name_basic_equal",
    "name_core_equal",
    "name_token_sorted_equal",

    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",

    "address_basic_equal",
    "address_canonical_equal",
    "address_token_sorted_equal",

    "address_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_jaccard",
    "canonical_jaccard",

    "shared_numbers",
    "shared_numbers_binary",

    "shared_postal",
    "shared_postal_binary",

    "first_number_equal",

    "country_equal",
    "address_missing_either",

    "is_s2",
    "is_s3",
]


print("\n" + "=" * 90)
print("FEATURE MEANS: POSITIVE VS NEGATIVE")
print("=" * 90)

summary_rows = []

for feature in FEATURES:

    if feature not in df.columns:
        continue

    positive_mean = (
        df.loc[
            df["label"] == 1,
            feature
        ].mean()
    )

    negative_mean = (
        df.loc[
            df["label"] == 0,
            feature
        ].mean()
    )

    summary_rows.append({
        "feature": feature,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "difference": (
            positive_mean
            - negative_mean
        )
    })


feature_summary = pd.DataFrame(
    summary_rows
).sort_values(
    "difference",
    ascending=False
)

print(
    feature_summary.to_string(
        index=False
    )
)

feature_summary.to_csv(
    OUTPUT_DIR
    / "feature_comparison.tsv",
    sep="\t",
    index=False
)


# =========================================================
# FALSE POSITIVE INSPECTION
# =========================================================

print("\n" + "=" * 90)
print("TOP FALSE POSITIVES")
print("=" * 90)

fp_display = false_positives.sort_values(
    "probability",
    ascending=False
).head(30)

for _, row in fp_display.iterrows():

    print(
        f"\nS1: {row['source1_entity_id']}"
    )

    print(
        f"Target: {row['target_entity_id']}"
    )

    print(
        f"Probability: "
        f"{row['probability']:.4f}"
    )

    print(
        f"Name TFIDF: "
        f"{row['name_tfidf']:.4f}"
    )

    print(
        f"Address TFIDF: "
        f"{row['address_tfidf']:.4f}"
    )

    print(
        f"Hybrid: "
        f"{row['hybrid_tfidf']:.4f}"
    )


# =========================================================
# FALSE NEGATIVE INSPECTION
# =========================================================

print("\n" + "=" * 90)
print("HARDEST FALSE NEGATIVES")
print("=" * 90)

fn_display = false_negatives.sort_values(
    "probability",
    ascending=False
).head(30)

for _, row in fn_display.iterrows():

    print(
        f"\nS1: {row['source1_entity_id']}"
    )

    print(
        f"Target: {row['target_entity_id']}"
    )

    print(
        f"Probability: "
        f"{row['probability']:.4f}"
    )

    print(
        f"Name TFIDF: "
        f"{row['name_tfidf']:.4f}"
    )

    print(
        f"Address TFIDF: "
        f"{row['address_tfidf']:.4f}"
    )

    print(
        f"Hybrid: "
        f"{row['hybrid_tfidf']:.4f}"
    )

    print(
        f"Shared numbers: "
        f"{row['shared_numbers']}"
    )

    print(
        f"Address Jaccard: "
        f"{row['address_jaccard']:.4f}"
    )


# =========================================================
# SCORE DISTRIBUTIONS
# =========================================================

print("\n" + "=" * 90)
print("PROBABILITY DISTRIBUTION")
print("=" * 90)

print("\nTrue matches:")

print(
    df.loc[
        df["label"] == 1,
        "probability"
    ].describe()
)

print("\nFalse matches:")

print(
    df.loc[
        df["label"] == 0,
        "probability"
    ].describe()
)


# =========================================================
# SAVE ERROR TABLES
# =========================================================

false_positives.to_csv(
    OUTPUT_DIR
    / "false_positives.tsv",
    sep="\t",
    index=False
)

false_negatives.to_csv(
    OUTPUT_DIR
    / "false_negatives.tsv",
    sep="\t",
    index=False
)

true_positives.to_csv(
    OUTPUT_DIR
    / "true_positives.tsv",
    sep="\t",
    index=False
)


print("\n" + "=" * 90)
print("ERROR ANALYSIS COMPLETE")
print("=" * 90)

print(
    f"Saved to: {OUTPUT_DIR}"
)