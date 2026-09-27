import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PATHS
# =========================================================

ROOT = Path(
    "ml challenge dataset/student_resource"
)

TRAIN = ROOT / "dataset/train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

OUT_DIR = Path("experiments")
OUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUT_DIR / "candidate_pairs_pilot.tsv"


# =========================================================
# SETTINGS
# =========================================================

NUM_QUERIES = 2000

DISTRACTORS_S2 = 100_000
DISTRACTORS_S3 = 100_000

SEED = 42

# TF-IDF retrieval depth
TFIDF_TOP_K = 500

# Maximum negatives retained per S1 after
# combining exact blocks + TF-IDF retrieval.
MAX_NEGATIVES_PER_S1 = 100

NAME_WEIGHT = 0.70
ADDRESS_WEIGHT = 0.30


# =========================================================
# NORMALIZATION
# =========================================================

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "llc",
    "ltd",
    "limited",
    "co",
    "company",
    "plc",
    "pvt",
    "private",
    "llp",
    "lp",
    "sarl",
    "sa",
    "sas",
    "gmbh",
    "ag",
    "bv",
    "srl",
    "spa",
    "pte",
}


ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "rdg": "ridge",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "pky": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "trl": "trail",
    "expy": "expressway",
    "fwy": "freeway",
    "sq": "square",
    "aly": "alley",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "ste": "suite",
    "mt": "mount",
}


def normalize(text):

    if not text:
        return ""

    text = str(text).casefold()

    chars = []

    for ch in text:

        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")

    return " ".join("".join(chars).split())


def normalize_name(name):

    text = normalize(name)

    if not text:
        return {
            "basic": "",
            "core": "",
            "token_sorted": "",
        }

    tokens = text.split()

    core_tokens = list(tokens)

    while (
        core_tokens
        and core_tokens[-1] in LEGAL_SUFFIXES
    ):
        core_tokens.pop()

    return {
        "basic": text,
        "core": " ".join(core_tokens),
        "token_sorted": " ".join(
            sorted(tokens)
        ),
    }


def normalize_address(address):

    basic = normalize(address)

    if not basic:
        return {
            "basic": "",
            "canonical": "",
            "token_sorted": "",
        }

    tokens = basic.split()

    canonical_tokens = [
        ADDRESS_ABBREVIATIONS.get(
            token,
            token
        )
        for token in tokens
    ]

    canonical = " ".join(
        canonical_tokens
    )

    token_sorted = " ".join(
        sorted(canonical_tokens)
    )

    return {
        "basic": basic,
        "canonical": canonical,
        "token_sorted": token_sorted,
    }


# =========================================================
# FILE HELPERS
# =========================================================

def reservoir_sample(path, k, seed=42):

    rng = random.Random(seed)

    sample = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for i, row in enumerate(reader):

            if i < k:
                sample.append(row)

            else:

                j = rng.randint(
                    0,
                    i
                )

                if j < k:
                    sample[j] = row

    return sample


def load_records(
    path,
    required_ids
):

    result = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if row["entity_id"] in required_ids:
                result[row["entity_id"]] = row

    return result


def sample_distractors(
    path,
    excluded_ids,
    k,
    seed
):

    rng = random.Random(seed)

    selected = []

    seen = 0

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if row["entity_id"] in excluded_ids:
                continue

            seen += 1

            if len(selected) < k:

                selected.append(row)

            else:

                j = rng.randint(
                    0,
                    seen - 1
                )

                if j < k:
                    selected[j] = row

    return selected


# =========================================================
# GROUND TRUTH
# =========================================================

print("=" * 90)
print("HYBRID CANDIDATE PAIR BUILDER")
print("=" * 90)

print("\nSampling ground truth...")

gt_sample = reservoir_sample(
    GT_FILE,
    NUM_QUERIES,
    SEED
)

true_by_s1 = {}

all_true_ids = set()

for row in gt_sample:

    s1_id = row["source1_entity_id"]

    value = row[
        "matched_entity_ids"
    ].strip()

    ids = (
        [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]
        if value
        else []
    )

    true_by_s1[s1_id] = set(ids)

    all_true_ids.update(ids)


s1_ids = set(
    true_by_s1.keys()
)

print(
    f"S1 queries       : {len(s1_ids):,}"
)

print(
    f"True target IDs  : {len(all_true_ids):,}"
)


# =========================================================
# LOAD S1
# =========================================================

print("\nLoading S1...")

s1 = load_records(
    S1_FILE,
    s1_ids
)


# =========================================================
# LOAD TRUE TARGETS
# =========================================================

print("Loading true S2 targets...")

true_s2 = load_records(
    S2_FILE,
    {
        x
        for x in all_true_ids
        if x.startswith("S2-")
    }
)

print("Loading true S3 targets...")

true_s3 = load_records(
    S3_FILE,
    {
        x
        for x in all_true_ids
        if x.startswith("S3-")
    }
)


# =========================================================
# DISTRACTORS
# =========================================================

print("\nSampling distractors...")

s2_distractors = sample_distractors(
    S2_FILE,
    all_true_ids,
    DISTRACTORS_S2,
    SEED
)

print(
    f"S2 distractors: "
    f"{len(s2_distractors):,}"
)

s3_distractors = sample_distractors(
    S3_FILE,
    all_true_ids,
    DISTRACTORS_S3,
    SEED + 1
)

print(
    f"S3 distractors: "
    f"{len(s3_distractors):,}"
)


# =========================================================
# TARGET POOL
# =========================================================

target_by_id = {}

for row in true_s2.values():
    target_by_id[row["entity_id"]] = row

for row in true_s3.values():
    target_by_id[row["entity_id"]] = row

for row in s2_distractors:
    target_by_id[row["entity_id"]] = row

for row in s3_distractors:
    target_by_id[row["entity_id"]] = row


target_records = list(
    target_by_id.values()
)

target_ids = [
    row["entity_id"]
    for row in target_records
]

target_index = {
    entity_id: i
    for i, entity_id in enumerate(
        target_ids
    )
}


print(
    f"\nTarget pool: "
    f"{len(target_records):,}"
)


# =========================================================
# PREPARE NORMALIZED DATA
# =========================================================

print("\nNormalizing target pool...")

target_name = []
target_address = []
target_country = []

for row in target_records:

    n = normalize_name(
        row["business_name"]
    )

    a = normalize_address(
        row["business_address"]
    )

    target_name.append(
        n
    )

    target_address.append(
        a
    )

    target_country.append(
        row["country"].casefold().strip()
    )


query_name = []
query_address = []
query_country = []

query_ids = list(s1.keys())

for s1_id in query_ids:

    row = s1[s1_id]

    n = normalize_name(
        row["business_name"]
    )

    a = normalize_address(
        row["business_address"]
    )

    query_name.append(n)
    query_address.append(a)

    query_country.append(
        row["country"].casefold().strip()
    )


# =========================================================
# EXACT BLOCK INDEXES
# =========================================================

print("\nBuilding exact blocking indexes...")

block_indexes = {
    "name_basic": defaultdict(set),
    "name_core": defaultdict(set),
    "name_token_sorted": defaultdict(set),
    "address_canonical": defaultdict(set),
    "address_token_sorted": defaultdict(set),
}


for i in range(
    len(target_records)
):

    country = target_country[i]
    n = target_name[i]
    a = target_address[i]

    if n["basic"]:
        block_indexes[
            "name_basic"
        ][
            (country, n["basic"])
        ].add(i)

    if n["core"]:
        block_indexes[
            "name_core"
        ][
            (country, n["core"])
        ].add(i)

    if n["token_sorted"]:
        block_indexes[
            "name_token_sorted"
        ][
            (country, n["token_sorted"])
        ].add(i)

    if a["canonical"]:
        block_indexes[
            "address_canonical"
        ][
            (country, a["canonical"])
        ].add(i)

    if a["token_sorted"]:
        block_indexes[
            "address_token_sorted"
        ][
            (country, a["token_sorted"])
        ].add(i)


# =========================================================
# TF-IDF
# =========================================================

print("\nFitting name TF-IDF...")

name_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 3),
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
)

target_name_text = [
    x["core"]
    for x in target_name
]

query_name_text = [
    x["core"]
    for x in query_name
]

target_name_matrix = (
    name_vectorizer.fit_transform(
        target_name_text
    )
)

query_name_matrix = (
    name_vectorizer.transform(
        query_name_text
    )
)


print(
    "Name matrix:",
    target_name_matrix.shape
)


print("\nFitting address TF-IDF...")

address_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b\w+\b",
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
)

target_address_text = [
    x["basic"]
    for x in target_address
]

query_address_text = [
    x["basic"]
    for x in query_address
]

target_address_matrix = (
    address_vectorizer.fit_transform(
        target_address_text
    )
)

query_address_matrix = (
    address_vectorizer.transform(
        query_address_text
    )
)


print(
    "Address matrix:",
    target_address_matrix.shape
)


# =========================================================
# RETRIEVAL
# =========================================================

print("\nGenerating candidate pairs...")

TARGET_CHUNK = 20_000

all_candidate_rows = []

start_total = time.time()


for qi, s1_id in enumerate(
    query_ids
):

    country = query_country[qi]

    # -----------------------------------------------------
    # Exact candidate union
    # -----------------------------------------------------

    candidates = set()

    n = query_name[qi]
    a = query_address[qi]

    exact_queries = [
        (
            "name_basic",
            n["basic"]
        ),
        (
            "name_core",
            n["core"]
        ),
        (
            "name_token_sorted",
            n["token_sorted"]
        ),
        (
            "address_canonical",
            a["canonical"]
        ),
        (
            "address_token_sorted",
            a["token_sorted"]
        ),
    ]

    for block_name, value in exact_queries:

        if not value:
            continue

        candidates.update(
            block_indexes[block_name].get(
                (country, value),
                set()
            )
        )

    # -----------------------------------------------------
    # TF-IDF top K
    # -----------------------------------------------------

    best_scores = {}

    q_name = query_name_matrix[qi]
    q_address = query_address_matrix[qi]

    for t_start in range(
        0,
        len(target_records),
        TARGET_CHUNK
    ):

        t_end = min(
            t_start + TARGET_CHUNK,
            len(target_records)
        )

        ns = (
            q_name
            @ target_name_matrix[
                t_start:t_end
            ].T
        ).toarray().ravel()

        ads = (
            q_address
            @ target_address_matrix[
                t_start:t_end
            ].T
        ).toarray().ravel()

        # Country filter
        for local_i in range(
            t_end - t_start
        ):

            global_i = (
                t_start + local_i
            )

            if (
                target_country[global_i]
                != country
            ):
                ns[local_i] = -np.inf
                ads[local_i] = -np.inf

        hybrid = (
            NAME_WEIGHT * ns
            +
            ADDRESS_WEIGHT * ads
        )

        # Keep only chunk top K
        if len(hybrid) > TFIDF_TOP_K:

            local_idx = np.argpartition(
                -hybrid,
                TFIDF_TOP_K
            )[:TFIDF_TOP_K]

        else:

            local_idx = np.arange(
                len(hybrid)
            )

        for local_i in local_idx:

            score = hybrid[local_i]

            if not np.isfinite(score):
                continue

            global_i = (
                t_start + local_i
            )

            best_scores[global_i] = (
                float(score)
            )

    # -----------------------------------------------------
    # Add all TF-IDF candidates
    # -----------------------------------------------------

    ranked_tfidf = sorted(
        best_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:TFIDF_TOP_K]

    for target_i, score in ranked_tfidf:
        candidates.add(target_i)

    # -----------------------------------------------------
    # Ensure every true positive from the pilot is present
    # in the candidate set.
    #
    # This does NOT affect model inference later.
    # It is specifically for training-pair construction,
    # because otherwise missed retrieval positives become
    # impossible for the classifier to learn.
    # -----------------------------------------------------

    true_ids = true_by_s1[s1_id]

    for true_id in true_ids:

        if true_id in target_index:
            candidates.add(
                target_index[true_id]
            )

    # -----------------------------------------------------
    # Score all candidates using hybrid similarity.
    # Exact-block candidates need explicit scoring.
    # -----------------------------------------------------

    candidate_list = list(
        candidates
    )

    scored = []

    for target_i in candidate_list:

        # Sparse dot products for one candidate.
        name_score = float(
            (
                q_name
                @ target_name_matrix[target_i].T
            ).toarray()[0, 0]
        )

        address_score = float(
            (
                q_address
                @ target_address_matrix[target_i].T
            ).toarray()[0, 0]
        )

        hybrid_score = (
            NAME_WEIGHT * name_score
            +
            ADDRESS_WEIGHT * address_score
        )

        scored.append(
            (
                target_i,
                name_score,
                address_score,
                hybrid_score
            )
        )

    # Put strongest candidates first.
    scored.sort(
        key=lambda x: x[3],
        reverse=True
    )

    # Keep ALL positives + strongest negatives.
    positive_rows = []
    negative_rows = []

    for (
        target_i,
        name_score,
        address_score,
        hybrid_score
    ) in scored:

        target_id = target_ids[target_i]

        label = (
            1
            if target_id in true_ids
            else 0
        )

        result = {
            "source1_entity_id": s1_id,
            "target_entity_id": target_id,
            "target_source": target_id[:2],

            "name_tfidf_score": name_score,
            "address_tfidf_score": address_score,
            "hybrid_tfidf_score": hybrid_score,

            "label": label,
        }

        if label == 1:
            positive_rows.append(result)
        else:
            negative_rows.append(result)

    # Keep hard negatives only.
    negative_rows = negative_rows[
        :MAX_NEGATIVES_PER_S1
    ]

    all_candidate_rows.extend(
        positive_rows
    )

    all_candidate_rows.extend(
        negative_rows
    )

    if (qi + 1) % 100 == 0:

        print(
            f"Processed "
            f"{qi + 1:,}/"
            f"{len(query_ids):,}"
        )


# =========================================================
# SAVE
# =========================================================

print("\nWriting candidate pairs...")

fieldnames = [
    "source1_entity_id",
    "target_entity_id",
    "target_source",
    "name_tfidf_score",
    "address_tfidf_score",
    "hybrid_tfidf_score",
    "label",
]


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        delimiter="\t"
    )

    writer.writeheader()

    writer.writerows(
        all_candidate_rows
    )


# =========================================================
# SUMMARY
# =========================================================

positives = sum(
    row["label"] == 1
    for row in all_candidate_rows
)

negatives = sum(
    row["label"] == 0
    for row in all_candidate_rows
)

print("\n" + "=" * 90)
print("CANDIDATE PAIR SUMMARY")
print("=" * 90)

print(
    f"Total pairs : {len(all_candidate_rows):,}"
)

print(
    f"Positive    : {positives:,}"
)

print(
    f"Negative    : {negatives:,}"
)

if all_candidate_rows:

    print(
        f"Positive rate: "
        f"{positives / len(all_candidate_rows):.4%}"
    )

print(
    f"\nTotal time: "
    f"{(time.time() - start_total) / 60:.2f} min"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("\nDone.")