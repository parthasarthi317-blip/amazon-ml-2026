import csv
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
)


# =========================================================
# PATHS
# =========================================================

TRAIN_FILE = Path(
    "experiments/pair_features_train.tsv"
)

VALID_FILE = Path(
    "experiments/pair_features_valid.tsv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "lightgbm_pair_matcher.txt"


# =========================================================
# SETTINGS
# =========================================================

RANDOM_STATE = 42

THRESHOLDS = np.arange(
    0.01,
    1.00,
    0.01
)


# =========================================================
# F0.5
# =========================================================

def f05_score(
    precision,
    recall
):
    """
    F0.5 gives more weight to precision.
    """

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


def macro_entity_f05(
    df,
    probabilities,
    threshold
):
    """
    Calculate macro F0.5 over S1 entities.

    For each S1:
        true set   = ground-truth target IDs
        predicted  = candidate target IDs whose probability
                     is >= threshold

    Then average the per-S1 F0.5.

    No-match + no-prediction receives F0.5 = 1.
    """

    work = df[
        [
            "source1_entity_id",
            "target_entity_id",
            "label",
        ]
    ].copy()

    work["probability"] = probabilities

    scores = []

    precision_values = []
    recall_values = []

    predicted_match_count = 0
    true_match_count = 0

    for s1_id, group in work.groupby(
        "source1_entity_id",
        sort=False
    ):

        true_ids = set(
            group.loc[
                group["label"] == 1,
                "target_entity_id"
            ]
        )

        predicted_ids = set(
            group.loc[
                group["probability"] >= threshold,
                "target_entity_id"
            ]
        )

        true_match_count += len(true_ids)
        predicted_match_count += len(
            predicted_ids
        )

        tp = len(
            true_ids & predicted_ids
        )

        fp = len(
            predicted_ids - true_ids
        )

        fn = len(
            true_ids - predicted_ids
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

        score = f05_score(
            precision,
            recall
        )

        scores.append(score)
        precision_values.append(
            precision
        )
        recall_values.append(
            recall
        )

    return {
        "macro_f05": (
            float(np.mean(scores))
            if scores
            else 0.0
        ),
        "macro_precision": (
            float(np.mean(precision_values))
            if precision_values
            else 0.0
        ),
        "macro_recall": (
            float(np.mean(recall_values))
            if recall_values
            else 0.0
        ),
        "predicted_matches": (
            predicted_match_count
        ),
        "true_matches": (
            true_match_count
        ),
    }


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 90)
print("LIGHTGBM PAIR MATCHER")
print("=" * 90)

print("\nLoading training features...")

train_df = pd.read_csv(
    TRAIN_FILE,
    sep="\t"
)

print(
    f"Train rows: "
    f"{len(train_df):,}"
)

print("\nLoading validation features...")

valid_df = pd.read_csv(
    VALID_FILE,
    sep="\t"
)

print(
    f"Validation rows: "
    f"{len(valid_df):,}"
)


# =========================================================
# FEATURES
# =========================================================

ID_COLUMNS = [
    "source1_entity_id",
    "target_entity_id",
]

TARGET_COLUMN = "label"

FEATURE_COLUMNS = [
    column
    for column in train_df.columns
    if column not in ID_COLUMNS
    and column != TARGET_COLUMN
]


print(
    f"\nNumber of features: "
    f"{len(FEATURE_COLUMNS)}"
)

print("\nFeatures:")

for feature in FEATURE_COLUMNS:
    print(
        f"  - {feature}"
    )


# =========================================================
# PREPARE MATRICES
# =========================================================

X_train = train_df[
    FEATURE_COLUMNS
].astype(np.float32)

y_train = train_df[
    TARGET_COLUMN
].astype(int)

X_valid = valid_df[
    FEATURE_COLUMNS
].astype(np.float32)

y_valid = valid_df[
    TARGET_COLUMN
].astype(int)


print("\nClass distribution:")

print(
    f"Train positives: "
    f"{int(y_train.sum()):,}"
)

print(
    f"Train negatives: "
    f"{int((y_train == 0).sum()):,}"
)

print(
    f"Valid positives: "
    f"{int(y_valid.sum()):,}"
)

print(
    f"Valid negatives: "
    f"{int((y_valid == 0).sum()):,}"
)


# =========================================================
# LIGHTGBM
# =========================================================

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=1200,

    learning_rate=0.04,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=40,

    subsample=0.85,

    colsample_bytree=0.85,

    reg_alpha=0.10,

    reg_lambda=1.00,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1,
)


model.fit(

    X_train,
    y_train,

    eval_set=[
        (
            X_valid,
            y_valid
        )
    ],

    eval_metric="auc",

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=75,
            verbose=True
        ),
        lgb.log_evaluation(
            period=50
        ),
    ],
)


# =========================================================
# SAVE MODEL
# =========================================================

print("\nSaving model...")

model.booster_.save_model(
    str(MODEL_FILE)
)

print(
    f"Saved: {MODEL_FILE}"
)


# =========================================================
# RAW VALIDATION PROBABILITIES
# =========================================================

print("\nPredicting validation set...")

valid_probabilities = model.predict_proba(
    X_valid
)[:, 1]


# =========================================================
# PAIRWISE METRICS
# =========================================================

auc = roc_auc_score(
    y_valid,
    valid_probabilities
)

print("\n" + "=" * 90)
print("PAIRWISE VALIDATION")
print("=" * 90)

print(
    f"ROC-AUC: {auc:.6f}"
)


# =========================================================
# THRESHOLD SEARCH
# =========================================================

print("\nSearching threshold for macro F0.5...")

results = []

for threshold in THRESHOLDS:

    metrics = macro_entity_f05(
        valid_df,
        valid_probabilities,
        float(threshold)
    )

    results.append({

        "threshold":
            float(threshold),

        "macro_f05":
            metrics["macro_f05"],

        "macro_precision":
            metrics["macro_precision"],

        "macro_recall":
            metrics["macro_recall"],

        "predicted_matches":
            metrics["predicted_matches"],

        "true_matches":
            metrics["true_matches"],
    })


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "macro_f05",
    ascending=False
)


# =========================================================
# BEST THRESHOLD
# =========================================================

best = results_df.iloc[0]


print("\n" + "=" * 90)
print("BEST THRESHOLD")
print("=" * 90)

print(
    f"Threshold        : "
    f"{best['threshold']:.2f}"
)

print(
    f"Macro F0.5       : "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Macro precision  : "
    f"{best['macro_precision']:.6f}"
)

print(
    f"Macro recall     : "
    f"{best['macro_recall']:.6f}"
)

print(
    f"Predicted matches: "
    f"{int(best['predicted_matches']):,}"
)

print(
    f"True matches     : "
    f"{int(best['true_matches']):,}"
)


# =========================================================
# TOP THRESHOLDS
# =========================================================

print("\nTop threshold candidates:")

print(
    results_df.head(15).to_string(
        index=False
    )
)


# =========================================================
# SAVE THRESHOLD RESULTS
# =========================================================

threshold_file = (
    MODEL_DIR
    / "threshold_search.tsv"
)

results_df.to_csv(
    threshold_file,
    sep="\t",
    index=False
)

print(
    f"\nSaved threshold results: "
    f"{threshold_file}"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({

    "feature":
        FEATURE_COLUMNS,

    "importance_gain":
        model.booster_.feature_importance(
            importance_type="gain"
        ),

    "importance_split":
        model.booster_.feature_importance(
            importance_type="split"
        ),
})

importance = importance.sort_values(
    "importance_gain",
    ascending=False
)

importance_file = (
    MODEL_DIR
    / "feature_importance.tsv"
)

importance.to_csv(
    importance_file,
    sep="\t",
    index=False
)

print(
    f"Saved feature importance: "
    f"{importance_file}"
)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 90)
print("TRAINING COMPLETE")
print("=" * 90)

print(
    f"Best validation Macro F0.5: "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Best threshold: "
    f"{best['threshold']:.2f}"
)

print(
    f"Model: {MODEL_FILE}"
)

print("\nDone.")