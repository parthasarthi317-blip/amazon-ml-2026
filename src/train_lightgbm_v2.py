import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score


# =========================================================
# PATHS
# =========================================================

TRAIN_FILE = Path(
    "experiments/pair_features_v2_train.tsv"
)

VALID_FILE = Path(
    "experiments/pair_features_v2_valid.tsv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = (
    MODEL_DIR
    / "lightgbm_pair_matcher_v2.txt"
)

THRESHOLD_FILE = (
    MODEL_DIR
    / "threshold_search_v2.tsv"
)

IMPORTANCE_FILE = (
    MODEL_DIR
    / "feature_importance_v2.tsv"
)


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

def f05(precision, recall):

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


def evaluate_entity_f05(
    df,
    probabilities,
    threshold
):

    work = df[
        [
            "source1_entity_id",
            "target_entity_id",
            "label",
        ]
    ].copy()

    work["probability"] = probabilities

    scores = []
    precisions = []
    recalls = []

    predicted_total = 0
    true_total = 0

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
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)

        score = f05(
            precision,
            recall
        )

        scores.append(score)
        precisions.append(precision)
        recalls.append(recall)

        predicted_total += len(
            predicted_ids
        )

        true_total += len(
            true_ids
        )

    return {
        "macro_f05": float(
            np.mean(scores)
        ),
        "macro_precision": float(
            np.mean(precisions)
        ),
        "macro_recall": float(
            np.mean(recalls)
        ),
        "predicted_matches":
            predicted_total,
        "true_matches":
            true_total,
    }


# =========================================================
# LOAD
# =========================================================

print("=" * 90)
print("LIGHTGBM V2")
print("=" * 90)

print("\nLoading train...")

train_df = pd.read_csv(
    TRAIN_FILE,
    sep="\t"
)

print(
    f"Train rows: "
    f"{len(train_df):,}"
)

print("\nLoading validation...")

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

TARGET = "label"

FEATURE_COLUMNS = [
    c
    for c in train_df.columns
    if c not in ID_COLUMNS
    and c != TARGET
]


print(
    f"\nFeature count: "
    f"{len(FEATURE_COLUMNS)}"
)


# =========================================================
# MATRICES
# =========================================================

X_train = train_df[
    FEATURE_COLUMNS
].astype(np.float32)

y_train = train_df[
    TARGET
].astype(np.int8)

X_valid = valid_df[
    FEATURE_COLUMNS
].astype(np.float32)

y_valid = valid_df[
    TARGET
].astype(np.int8)


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
# MODEL
# =========================================================

print("\nTraining LightGBM V2...")

model = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=1600,

    learning_rate=0.035,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=40,

    subsample=0.85,

    colsample_bytree=0.90,

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
            stopping_rounds=100,
            verbose=True
        ),
        lgb.log_evaluation(
            period=50
        ),
    ],
)


# =========================================================
# SAVE
# =========================================================

model.booster_.save_model(
    str(MODEL_FILE)
)

print(
    f"\nModel saved: {MODEL_FILE}"
)


# =========================================================
# PREDICT
# =========================================================

print("\nPredicting validation...")

probabilities = model.predict_proba(
    X_valid
)[:, 1]


auc = roc_auc_score(
    y_valid,
    probabilities
)

print(
    f"Pairwise ROC-AUC: "
    f"{auc:.6f}"
)


# =========================================================
# THRESHOLD SEARCH
# =========================================================

print("\nSearching thresholds...")

results = []

for threshold in THRESHOLDS:

    metrics = evaluate_entity_f05(
        valid_df,
        probabilities,
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
).sort_values(
    "macro_f05",
    ascending=False
)


best = results_df.iloc[0]


print("\n" + "=" * 90)
print("V2 BEST THRESHOLD")
print("=" * 90)

print(
    f"Threshold       : "
    f"{best['threshold']:.2f}"
)

print(
    f"Macro F0.5      : "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Macro precision : "
    f"{best['macro_precision']:.6f}"
)

print(
    f"Macro recall    : "
    f"{best['macro_recall']:.6f}"
)

print(
    f"Predicted       : "
    f"{int(best['predicted_matches']):,}"
)

print(
    f"True            : "
    f"{int(best['true_matches']):,}"
)


print("\nTop thresholds:")

print(
    results_df.head(15).to_string(
        index=False
    )
)


# =========================================================
# SAVE THRESHOLDS
# =========================================================

results_df.to_csv(
    THRESHOLD_FILE,
    sep="\t",
    index=False
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

importance.to_csv(
    IMPORTANCE_FILE,
    sep="\t",
    index=False
)


print(
    f"\nThreshold results saved: "
    f"{THRESHOLD_FILE}"
)

print(
    f"Feature importance saved: "
    f"{IMPORTANCE_FILE}"
)


# =========================================================
# TOP FEATURES
# =========================================================

print("\n" + "=" * 90)
print("TOP 20 FEATURES BY GAIN")
print("=" * 90)

print(
    importance.head(20).to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("V2 TRAINING COMPLETE")
print("=" * 90)

print(
    f"Best Macro F0.5: "
    f"{best['macro_f05']:.6f}"
)

print(
    f"Best threshold: "
    f"{best['threshold']:.2f}"
)

print("\nDone.")