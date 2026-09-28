import gc
import hashlib
import heapq
import os
from collections import defaultdict

import anyascii
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from normalization import (
    normalize_name_v2,
    normalize_name_token_sorted,
    normalize_address,
    extract_address_components,
)

# We reuse the blocking implementation already developed
# in the repository.
from signal_ranked_blocking import (
    read_chunks,
    build_token_sorted_name_index,
    build_name_token_index,
    build_address_token_index,
    build_exact_address_index,
    score_candidates,
    add_broad_fallback_scores,
)


# ============================================================
# PATHS
# ============================================================

ROOT = os.getcwd()

TRAIN_DIR = os.path.join(
    ROOT,
    "dataset",
    "train",
)

S1_FILE = os.path.join(
    TRAIN_DIR,
    "train_source1.tsv",
)

S2_FILE = os.path.join(
    TRAIN_DIR,
    "train_source2.tsv",
)

S3_FILE = os.path.join(
    TRAIN_DIR,
    "train_source3.tsv",
)

GT_FILE = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv",
)

MODEL_DIR = os.path.join(
    ROOT,
    "models",
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "competitive_lgbm.joblib",
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_S1_LIMIT = 100_000

TRAIN_TOP_K = 40

CHUNK_SIZE = 100_000

MIN_NAME_TOKEN_LEN = 3
MIN_ADDRESS_TOKEN_LEN = 3

RANDOM_SEED = 42


# ============================================================
# FEATURE LIST
# ============================================================

FEATURE_NAMES = [
    # Name
    "name_exact",
    "name_sorted_exact",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_wratio",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_containment",
    "name_length_ratio",

    # Transliteration
    "name_translit_ratio",
    "name_translit_token_set_ratio",
    "name_translit_exact",

    # Address
    "address_exact",
    "address_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_wratio",
    "address_jaro_winkler",
    "address_token_jaccard",
    "address_token_containment",
    "address_length_ratio",

    # Numeric/address structure
    "number_overlap",
    "number_overlap_ratio",
    "postal_overlap",
    "house_number_match",
    "number_count_diff",

    # Context
    "country_match",
    "source3",
    "target_name_missing",
    "target_address_missing",

    # Script / structure
    "name_script_mismatch",
    "name_length_difference",

    # Blocking evidence
    "blocking_score",
    "blocking_score_ge_10",
    "blocking_score_ge_6",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_ratio(a, b):
    if not a or not b:
        return 0.0

    return min(
        len(a),
        len(b),
    ) / max(
        len(a),
        len(b),
    )


def token_jaccard(tokens_a, tokens_b):
    a = set(tokens_a)
    b = set(tokens_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(
        a & b
    ) / len(
        a | b
    )


def token_containment(tokens_a, tokens_b):
    """
    Fraction of the smaller token set contained in
    the larger token set.
    """

    a = set(tokens_a)
    b = set(tokens_b)

    if not a or not b:
        return 0.0

    return len(
        a & b
    ) / min(
        len(a),
        len(b),
    )


def number_features(address_a, address_b):

    a = extract_address_components(
        address_a
    )

    b = extract_address_components(
        address_b
    )

    numbers_a = set(
        a["numbers"]
    )

    numbers_b = set(
        b["numbers"]
    )

    postal_a = set(
        a["postal_codes"]
    )

    postal_b = set(
        b["postal_codes"]
    )

    if numbers_a and numbers_b:

        overlap = len(
            numbers_a
            & numbers_b
        )

        overlap_ratio = (
            overlap
            / min(
                len(numbers_a),
                len(numbers_b),
            )
        )

    else:

        overlap = 0
        overlap_ratio = 0.0

    postal_overlap = (
        1.0
        if postal_a
        and postal_b
        and postal_a & postal_b
        else 0.0
    )

    house_match = 0.0

    if numbers_a and numbers_b:

        house_match = (
            1.0
            if list(numbers_a)[0]
            == list(numbers_b)[0]
            else 0.0
        )

    number_count_diff = abs(
        len(numbers_a)
        - len(numbers_b)
    )

    return (
        float(overlap),
        float(overlap_ratio),
        float(postal_overlap),
        float(house_match),
        float(number_count_diff),
    )


def script_type(text):

    if not text:
        return "empty"

    has_ascii_latin = False
    has_non_latin = False

    for char in text:

        code = ord(char)

        if (
            65 <= code <= 90
            or 97 <= code <= 122
        ):

            has_ascii_latin = True

        elif char.isalpha():

            has_non_latin = True

    if has_ascii_latin and has_non_latin:
        return "mixed"

    if has_ascii_latin:
        return "latin"

    if has_non_latin:
        return "nonlatin"

    return "other"


def transliterate(text):

    if not text:
        return ""

    return anyascii.anyascii(
        text
    ).lower().strip()


# ============================================================
# SOURCE RECORD STORE
# ============================================================

def load_source_store(path):

    """
    Loads only the normalized fields required by the model.

    Returns:

        entity_id -> {
            country,
            name,
            sorted_name,
            translit_name,
            address,
        }
    """

    print(
        f"\nLoading feature store: "
        f"{os.path.basename(path)}"
    )

    store = {}

    for chunk_no, chunk in enumerate(
        read_chunks(path),
        start=1,
    ):

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

            raw_name = row.business_name
            raw_address = row.business_address

            name = (
                normalize_name_v2(
                    raw_name
                )
                if raw_name
                else ""
            )

            sorted_name = (
                normalize_name_token_sorted(
                    raw_name
                )
                if raw_name
                else ""
            )

            address = (
                normalize_address(
                    raw_address
                )
                if raw_address
                else ""
            )

            store[entity_id] = {
                "country": row.country,
                "name": name,
                "sorted_name": sorted_name,
                "translit_name": transliterate(
                    name
                ),
                "address": address,
            }

        print(
            f"  loaded chunk {chunk_no}"
        )

    print(
        f"Stored records: "
        f"{len(store):,}"
    )

    return store


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_pair_features(
    s1_country,
    s1_name_raw,
    s1_address_raw,
    candidate_ids,
    blocking_scores,
    target_store,
):

    name1 = (
        normalize_name_v2(
            s1_name_raw
        )
        if s1_name_raw
        else ""
    )

    sorted_name1 = (
        normalize_name_token_sorted(
            s1_name_raw
        )
        if s1_name_raw
        else ""
    )

    translit_name1 = transliterate(
        name1
    )

    address1 = (
        normalize_address(
            s1_address_raw
        )
        if s1_address_raw
        else ""
    )

    name_tokens1 = (
        name1.split()
        if name1
        else []
    )

    address_tokens1 = (
        address1.split()
        if address1
        else []
    )

    source3_flags = {}

    X = np.zeros(
        (
            len(candidate_ids),
            len(FEATURE_NAMES),
        ),
        dtype=np.float32,
    )

    for i, candidate_id in enumerate(
        candidate_ids
    ):

        target = target_store.get(
            candidate_id
        )

        if target is None:
            continue

        name2 = target["name"]
        sorted_name2 = target["sorted_name"]
        translit_name2 = target["translit_name"]
        address2 = target["address"]

        name_tokens2 = (
            name2.split()
            if name2
            else []
        )

        address_tokens2 = (
            address2.split()
            if address2
            else []
        )

        # ----------------------------------------------------
        # Name similarities
        # ----------------------------------------------------

        name_exact = (
            1.0
            if name1
            and name2
            and name1 == name2
            else 0.0
        )

        sorted_exact = (
            1.0
            if sorted_name1
            and sorted_name2
            and sorted_name1 == sorted_name2
            else 0.0
        )

        name_ratio = (
            fuzz.ratio(
                name1,
                name2,
            )
            / 100.0
            if name1
            and name2
            else 0.0
        )

        name_token_sort = (
            fuzz.token_sort_ratio(
                name1,
                name2,
            )
            / 100.0
            if name1
            and name2
            else 0.0
        )

        name_token_set = (
            fuzz.token_set_ratio(
                name1,
                name2,
            )
            / 100.0
            if name1
            and name2
            else 0.0
        )

        name_wratio = (
            fuzz.WRatio(
                name1,
                name2,
            )
            / 100.0
            if name1
            and name2
            else 0.0
        )

        name_jw = (
            JaroWinkler.normalized_similarity(
                name1,
                name2,
            )
            if name1
            and name2
            else 0.0
        )

        name_jaccard = token_jaccard(
            name_tokens1,
            name_tokens2,
        )

        name_containment = token_containment(
            name_tokens1,
            name_tokens2,
        )

        name_len_ratio = safe_ratio(
            name1,
            name2,
        )

        # ----------------------------------------------------
        # Transliteration
        # ----------------------------------------------------

        translit_ratio = (
            fuzz.ratio(
                translit_name1,
                translit_name2,
            )
            / 100.0
            if translit_name1
            and translit_name2
            else 0.0
        )

        translit_token_set = (
            fuzz.token_set_ratio(
                translit_name1,
                translit_name2,
            )
            / 100.0
            if translit_name1
            and translit_name2
            else 0.0
        )

        translit_exact = (
            1.0
            if translit_name1
            and translit_name2
            and translit_name1
            == translit_name2
            else 0.0
        )

        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        address_exact = (
            1.0
            if address1
            and address2
            and address1 == address2
            else 0.0
        )

        address_ratio = (
            fuzz.ratio(
                address1,
                address2,
            )
            / 100.0
            if address1
            and address2
            else 0.0
        )

        address_token_sort = (
            fuzz.token_sort_ratio(
                address1,
                address2,
            )
            / 100.0
            if address1
            and address2
            else 0.0
        )

        address_token_set = (
            fuzz.token_set_ratio(
                address1,
                address2,
            )
            / 100.0
            if address1
            and address2
            else 0.0
        )

        address_wratio = (
            fuzz.WRatio(
                address1,
                address2,
            )
            / 100.0
            if address1
            and address2
            else 0.0
        )

        address_jw = (
            JaroWinkler.normalized_similarity(
                address1,
                address2,
            )
            if address1
            and address2
            else 0.0
        )

        address_jaccard = token_jaccard(
            address_tokens1,
            address_tokens2,
        )

        address_containment = token_containment(
            address_tokens1,
            address_tokens2,
        )

        address_len_ratio = safe_ratio(
            address1,
            address2,
        )

        (
            number_overlap,
            number_overlap_ratio,
            postal_overlap,
            house_match,
            number_count_diff,
        ) = number_features(
            address1,
            address2,
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        country_match = (
            1.0
            if target["country"]
            == s1_country
            else 0.0
        )

        source3 = (
            1.0
            if candidate_id.startswith(
                "S3-"
            )
            else 0.0
        )

        target_name_missing = (
            1.0
            if not name2
            else 0.0
        )

        target_address_missing = (
            1.0
            if not address2
            else 0.0
        )

        script_mismatch = (
            1.0
            if (
                script_type(name1)
                != script_type(name2)
                and name1
                and name2
            )
            else 0.0
        )

        name_length_difference = abs(
            len(name1)
            - len(name2)
        )

        blocking_score = float(
            blocking_scores.get(
                candidate_id,
                0,
            )
        )

        X[i] = [
            name_exact,
            sorted_exact,
            name_ratio,
            name_token_sort,
            name_token_set,
            name_wratio,
            name_jw,
            name_jaccard,
            name_containment,
            name_len_ratio,

            translit_ratio,
            translit_token_set,
            translit_exact,

            address_exact,
            address_ratio,
            address_token_sort,
            address_token_set,
            address_wratio,
            address_jw,
            address_jaccard,
            address_containment,
            address_len_ratio,

            number_overlap,
            number_overlap_ratio,
            postal_overlap,
            house_match,
            number_count_diff,

            country_match,
            source3,
            target_name_missing,
            target_address_missing,

            script_mismatch,
            name_length_difference,

            blocking_score,
            1.0 if blocking_score >= 10 else 0.0,
            1.0 if blocking_score >= 6 else 0.0,
        ]

    return X


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(path):

    print(
        "\nLoading ground truth..."
    )

    gt = {}

    for chunk in read_chunks(path):

        for row in chunk.itertuples(
            index=False
        ):

            value = (
                row.matched_entity_ids
                .strip()
            )

            if value:

                gt[
                    row.source1_entity_id
                ] = {
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                }

            else:

                gt[
                    row.source1_entity_id
                ] = set()

    print(
        f"Ground truth records: "
        f"{len(gt):,}"
    )

    return gt


# ============================================================
# ENTITY F0.5
# ============================================================

def entity_f05(
    truth,
    prediction,
):

    truth = set(truth)
    prediction = set(prediction)

    if not truth:

        return (
            1.0
            if not prediction
            else 0.0
        )

    if not prediction:
        return 0.0

    tp = len(
        truth
        & prediction
    )

    if tp == 0:
        return 0.0

    precision = (
        tp
        / len(prediction)
    )

    recall = (
        tp
        / len(truth)
    )

    return (
        1.25
        * precision
        * recall
        / (
            0.25 * precision
            + recall
        )
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidate_pool(
    country,
    name,
    address,
    indexes,
):

    (
        name_sorted_s2,
        name_sorted_s3,
        name_tokens_s2,
        name_tokens_s3,
        address_tokens_s2,
        address_tokens_s3,
        exact_address_s2,
        exact_address_s3,
    ) = indexes

    scores = score_candidates(
        country,
        name,
        address,

        name_sorted_s2,
        name_sorted_s3,

        name_tokens_s2,
        name_tokens_s3,

        address_tokens_s2,
        address_tokens_s3,

        exact_address_s2,
        exact_address_s3,
    )

    # Same adaptive fallback used by E017.
    if len(scores) <= 20:

        add_broad_fallback_scores(
            scores,
            country,
            address,
            address_tokens_s2,
            address_tokens_s3,
        )

    if not scores:
        return {}

    ranked = heapq.nlargest(
        TRAIN_TOP_K,
        scores.items(),
        key=lambda x: (
            x[1],
            x[0],
        ),
    )

    return dict(ranked)


# ============================================================
# TRAIN
# ============================================================

def main():

    print("=" * 80)
    print("COMPETITIVE LIGHTGBM ENTITY MATCHER")
    print("=" * 80)

    print(
        f"\nTraining S1 limit: "
        f"{TRAIN_S1_LIMIT:,}"
    )

    print(
        f"Candidate top-K: "
        f"{TRAIN_TOP_K}"
    )

    # --------------------------------------------------------
    # BUILD BLOCKING INDEXES
    # --------------------------------------------------------

    print(
        "\n1. Building blocking indexes"
    )

    name_sorted_s2 = (
        build_token_sorted_name_index(
            S2_FILE
        )
    )

    name_sorted_s3 = (
        build_token_sorted_name_index(
            S3_FILE
        )
    )

    name_tokens_s2 = (
        build_name_token_index(
            S2_FILE
        )
    )

    name_tokens_s3 = (
        build_name_token_index(
            S3_FILE
        )
    )

    address_tokens_s2 = (
        build_address_token_index(
            S2_FILE
        )
    )

    address_tokens_s3 = (
        build_address_token_index(
            S3_FILE
        )
    )

    exact_address_s2 = (
        build_exact_address_index(
            S2_FILE
        )
    )

    exact_address_s3 = (
        build_exact_address_index(
            S3_FILE
        )
    )

    indexes = (
        name_sorted_s2,
        name_sorted_s3,
        name_tokens_s2,
        name_tokens_s3,
        address_tokens_s2,
        address_tokens_s3,
        exact_address_s2,
        exact_address_s3,
    )

    # --------------------------------------------------------
    # SOURCE FEATURE STORES
    # --------------------------------------------------------

    print(
        "\n2. Loading Source 2 feature store"
    )

    store_s2 = load_source_store(
        S2_FILE
    )

    print(
        "\n3. Loading Source 3 feature store"
    )

    store_s3 = load_source_store(
        S3_FILE
    )

    target_store = {}

    target_store.update(
        store_s2
    )

    target_store.update(
        store_s3
    )

    del store_s2
    del store_s3

    gc.collect()

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    gt = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # TRAINING RECORDS
    # --------------------------------------------------------

    training_rows = []

    validation_rows = []

    print(
        "\n4. Collecting training pairs"
    )

    seen = 0
    selected = 0

    for chunk in read_chunks(
        S1_FILE
    ):

        for row in chunk.itertuples(
            index=False
        ):

            if selected >= TRAIN_S1_LIMIT:
                break

            # Deterministic hash split.
            digest = hashlib.md5(
                row.entity_id.encode(
                    "utf-8"
                )
            ).digest()

            is_validation = (
                digest[0] % 5 == 0
            )

            candidates = (
                generate_candidate_pool(
                    row.country,
                    row.business_name,
                    row.business_address,
                    indexes,
                )
            )

            if not candidates:

                selected += 1
                continue

            candidate_ids = list(
                candidates.keys()
            )

            truth = gt.get(
                row.entity_id,
                set(),
            )

            positives = [
                cid
                for cid in candidate_ids
                if cid in truth
            ]

            negatives = [
                cid
                for cid in candidate_ids
                if cid not in truth
            ]

            # Hard negatives first.
            negatives.sort(
                key=lambda cid: (
                    -candidates[cid],
                    cid,
                )
            )

            negatives = negatives[
                :15
            ]

            # Make sure every positive in the pool is used.
            pair_ids = (
                positives
                + negatives
            )

            X = make_pair_features(
                row.country,
                row.business_name,
                row.business_address,
                pair_ids,
                candidates,
                target_store,
            )

            y = np.asarray(
                [
                    1
                    if cid in truth
                    else 0
                    for cid in pair_ids
                ],
                dtype=np.int8,
            )

            if is_validation:

                validation_rows.append(
                    {
                        "s1_id":
                            row.entity_id,
                        "truth":
                            truth,
                        "candidate_ids":
                            pair_ids,
                        "X":
                            X,
                    }
                )

            else:

                training_rows.append(
                    (
                        X,
                        y,
                    )
                )

            selected += 1

        print(
            f"Selected S1 records: "
            f"{selected:,}"
        )

        if selected >= TRAIN_S1_LIMIT:
            break

    # --------------------------------------------------------
    # STACK TRAINING DATA
    # --------------------------------------------------------

    print(
        "\n5. Stacking training matrix"
    )

    X_train = np.vstack(
        [
            x
            for x, _ in training_rows
        ]
    )

    y_train = np.concatenate(
        [
            y
            for _, y in training_rows
        ]
    )

    X_val = np.vstack(
        [
            r["X"]
            for r in validation_rows
        ]
    )

    y_val = np.concatenate(
        [
            np.asarray(
                [
                    1
                    if cid in r["truth"]
                    else 0
                    for cid in r[
                        "candidate_ids"
                    ]
                ],
                dtype=np.int8,
            )
            for r in validation_rows
        ]
    )

    print(
        f"Training pairs: "
        f"{len(y_train):,}"
    )

    print(
        f"Validation pairs: "
        f"{len(y_val):,}"
    )

    print(
        f"Training positives: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Validation positives: "
        f"{int(y_val.sum()):,}"
    )

    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    pos = max(
        1,
        int(y_train.sum()),
    )

    neg = max(
        1,
        int(
            (y_train == 0).sum()
        ),
    )

    scale_pos_weight = (
        neg / pos
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.3f}"
    )

    model = lgb.LGBMClassifier(
        objective="binary",

        n_estimators=800,
        learning_rate=0.035,

        num_leaves=63,
        max_depth=-1,

        min_child_samples=50,

        subsample=0.85,
        colsample_bytree=0.90,

        reg_alpha=0.25,
        reg_lambda=3.0,

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_SEED,

        n_jobs=-1,

        verbosity=-1,
    )

    print(
        "\n6. Training LightGBM"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_val,
                y_val,
            )
        ],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                60,
                verbose=True,
            ),
        ],
    )

    # --------------------------------------------------------
    # VALIDATION RAW PROBABILITIES
    # --------------------------------------------------------

    print(
        "\n7. Calibrating probabilities"
    )

    raw_val_prob = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    raw_auc = roc_auc_score(
        y_val,
        raw_val_prob,
    )

    raw_ap = average_precision_score(
        y_val,
        raw_val_prob,
    )

    print(
        f"Raw validation AUC: "
        f"{raw_auc:.6f}"
    )

    print(
        f"Raw validation AP: "
        f"{raw_ap:.6f}"
    )

    # --------------------------------------------------------
    # ISOTONIC CALIBRATION
    # --------------------------------------------------------

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        raw_val_prob,
        y_val,
    )

    calibrated_val_prob = (
        calibrator.predict(
            raw_val_prob
        )
    )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    print(
        "\n8. Tuning macro F0.5"
    )

    # Map validation probabilities back to entities.
    offset = 0

    val_predictions_by_threshold = {}

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    for threshold in thresholds:

        entity_scores = []

        offset = 0

        for record in validation_rows:

            count = len(
                record[
                    "candidate_ids"
                ]
            )

            probabilities = (
                calibrated_val_prob[
                    offset:
                    offset + count
                ]
            )

            offset += count

            predicted = [
                cid
                for cid, probability
                in zip(
                    record[
                        "candidate_ids"
                    ],
                    probabilities,
                )
                if probability
                >= threshold
            ]

            entity_scores.append(
                entity_f05(
                    record["truth"],
                    predicted,
                )
            )

        mean_f05 = (
            float(
                np.mean(
                    entity_scores
                )
            )
            if entity_scores
            else 0.0
        )

        val_predictions_by_threshold[
            threshold
        ] = mean_f05

        print(
            f"Threshold "
            f"{threshold:.2f}"
            f" -> Macro F0.5 "
            f"{mean_f05:.6f}"
        )

    best_threshold = max(
        val_predictions_by_threshold,
        key=val_predictions_by_threshold.get,
    )

    best_f05 = (
        val_predictions_by_threshold[
            best_threshold
        ]
    )

    print(
        "\nBEST THRESHOLD:"
        f" {best_threshold:.2f}"
    )

    print(
        "BEST VALIDATION MACRO F0.5:"
        f" {best_f05:.6f}"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print(
        "\n9. Feature importance"
    )

    importances = (
        model.feature_importances_
    )

    ranked_features = sorted(
        zip(
            FEATURE_NAMES,
            importances,
        ),
        key=lambda x: -x[1],
    )

    for name, importance in (
        ranked_features[:15]
    ):

        print(
            f"  {name:35s} "
            f"{importance}"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "calibrator": calibrator,
        "threshold": float(
            best_threshold
        ),
        "feature_names": FEATURE_NAMES,
        "train_top_k": TRAIN_TOP_K,
        "version": "COMPETITIVE-001",
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
        compress=3,
    )

    print(
        f"\n10. Saved model:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    del X_train
    del X_val
    del y_train
    del y_val
    del training_rows
    del validation_rows
    del target_store
    del gt

    gc.collect()

    print(
        "\n" + "=" * 80
    )

    print(
        "MODEL TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()