import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PRED_FILE = Path(
    "experiments/clean_v2_predictions.tsv"
)

GT_FILE = Path(
    "ml challenge dataset/student_resource/"
    "dataset/train/train_ground_truth.tsv"
)

OUTPUT_FILE = Path(
    "experiments/clean_v2_predictions_labeled.tsv"
)

THRESHOLD_FILE = Path(
    "models/clean_threshold_search_v2.tsv"
)


# =========================================================
# F0.5
# =========================================================

def f05(
    precision,
    recall
):

    beta = 0.5

    denominator = (
        beta * beta * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta * beta)
        * precision
        * recall
        / denominator
    )


# =========================================================
# LOAD PREDICTIONS
# =========================================================

print("=" * 90)
print("FINISH CLEAN V2 EVALUATION")
print("=" * 90)

print("\nLoading saved V2 predictions...")

df = pd.read_csv(
    PRED_FILE,
    sep="\t"
)

df["source1_entity_id"] = (
    df["source1_entity_id"].astype(str)
)

df["target_entity_id"] = (
    df["target_entity_id"].astype(str)
)

print(
    f"Prediction rows: {len(df):,}"
)

print(
    f"S1 entities: "
    f"{df['source1_entity_id'].nunique():,}"
)


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

validation_s1 = set(
    df["source1_entity_id"].unique()
)

gt = gt[
    gt["source1_entity_id"].isin(
        validation_s1
    )
]

true_by_s1 = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value):
        value = ""

    true_by_s1[
        row["source1_entity_id"]
    ] = {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


# Make sure S1s with no match exist.
for sid in validation_s1:

    true_by_s1.setdefault(
        sid,
        set()
    )


true_total = sum(
    len(x)
    for x in true_by_s1.values()
)

print(
    f"True matches: "
    f"{true_total:,}"
)


# =========================================================
# RECONSTRUCT LABEL
# =========================================================

print("\nReconstructing labels...")

df["label"] = [
    int(
        target_id
        in true_by_s1[s1_id]
    )
    for s1_id, target_id
    in zip(
        df["source1_entity_id"],
        df["target_entity_id"]
    )
]

print(
    f"Retrieved true pairs: "
    f"{int(df['label'].sum()):,}"
)


# =========================================================
# CANDIDATE RECALL
# =========================================================

print("\nCalculating candidate recall...")

pair_hits = 0
full_entity_hits = 0
matched_entities = 0

for sid, group in df.groupby(
    "source1_entity_id",
    sort=False
):

    truth = true_by_s1[sid]

    retrieved = set(
        group[
            "target_entity_id"
        ]
    )

    pair_hits += len(
        truth & retrieved
    )

    if truth:

        matched_entities += 1

        if truth.issubset(
            retrieved
        ):
            full_entity_hits += 1


pair_recall = (
    pair_hits / true_total
    if true_total
    else 0.0
)

entity_recall = (
    full_entity_hits / matched_entities
    if matched_entities
    else 0.0
)


print(
    f"Pair recall @1000       : "
    f"{pair_recall:.2%}"
)

print(
    f"Full entity recall @1000: "
    f"{entity_recall:.2%}"
)


# =========================================================
# PRECOMPUTE GROUPS
# =========================================================

print("\nPreparing fast grouped arrays...")

groups = []

for sid, group in df.groupby(
    "source1_entity_id",
    sort=False
):

    probs = (
        group["probability"]
        .to_numpy(
            dtype=np.float64
        )
    )

    labels = (
        group["label"]
        .to_numpy(
            dtype=np.int8
        )
    )

    order = np.argsort(
        -probs
    )

    probs = probs[order]
    labels = labels[order]

    cumulative_tp = np.cumsum(
        labels,
        dtype=np.int32
    )

    true_count = int(
        labels.sum()
    )

    groups.append(
        (
            probs,
            cumulative_tp,
            true_count
        )
    )


print(
    f"Prepared groups: "
    f"{len(groups):,}"
)


# =========================================================
# FAST EVALUATION
# =========================================================

def evaluate(
    threshold
):

    scores = []
    precisions = []
    recalls = []

    predicted_total = 0
    true_total_local = 0

    for (
        probs,
        cumulative_tp,
        true_count
    ) in groups:

        # probs sorted descending.
        # Search in reversed ascending array
        # to find count >= threshold.
        insertion = np.searchsorted(
            probs[::-1],
            threshold,
            side="left"
        )

        predicted_count = (
            len(probs) - insertion
        )

        if predicted_count > 0:

            tp = int(
                cumulative_tp[
                    predicted_count - 1
                ]
            )

        else:

            tp = 0

        fp = (
            predicted_count - tp
        )

        fn = (
            true_count - tp
        )

        if tp + fp == 0:

            precision = (
                1.0
                if tp + fn == 0
                else 0.0
            )

        else:

            precision = (
                tp / (tp + fp)
            )

        if tp + fn == 0:

            recall = 1.0

        else:

            recall = (
                tp / (tp + fn)
            )

        scores.append(
            f05(
                precision,
                recall
            )
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        predicted_total += (
            predicted_count
        )

        true_total_local += (
            true_count
        )

    return {
        "macro_f05":
            float(np.mean(scores)),

        "macro_precision":
            float(np.mean(precisions)),

        "macro_recall":
            float(np.mean(recalls)),

        "predicted_matches":
            predicted_total,

        "true_matches":
            true_total_local,
    }


# =========================================================
# THRESHOLD SWEEP
# =========================================================

print("\nSearching thresholds...")

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

results = []

for threshold in thresholds:

    result = evaluate(
        float(threshold)
    )

    results.append({
        "threshold":
            float(threshold),

        **result
    })


results_df = pd.DataFrame(
    results
).sort_values(
    "macro_f05",
    ascending=False
)

best = results_df.iloc[0]

# Explicit 0.58 result
at_058 = evaluate(
    0.58
)


# =========================================================
# OUTPUT
# =========================================================

print("\n" + "=" * 90)
print("CLEAN V2 RESULTS")
print("=" * 90)

print(
    f"Candidate pair recall : "
    f"{pair_recall:.2%}"
)

print(
    f"Candidate entity recall: "
    f"{entity_recall:.2%}"
)

print()

print(
    f"F0.5 @ 0.58           : "
    f"{at_058['macro_f05']:.6f}"
)

print(
    f"Best threshold        : "
    f"{best['threshold']:.2f}"
)

print(
    f"Best Macro F0.5       : "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Best precision        : "
    f"{best['macro_precision']:.6f}"
)

print(
    f"Best recall           : "
    f"{best['macro_recall']:.6f}"
)

print(
    f"Predicted matches     : "
    f"{int(best['predicted_matches']):,}"
)

print(
    f"True matches          : "
    f"{int(best['true_matches']):,}"
)


print("\nTop thresholds:")

print(
    results_df.head(15).to_string(
        index=False
    )
)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

results_df.to_csv(
    THRESHOLD_FILE,
    sep="\t",
    index=False
)

print("\nSaved:")
print(
    f"Predictions: "
    f"{OUTPUT_FILE}"
)

print(
    f"Thresholds : "
    f"{THRESHOLD_FILE}"
)

print("\nDone.")