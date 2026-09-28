import csv
import gc
import heapq
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression

from normalization import (
    normalize_name_v2,
    normalize_name_token_sorted,
    normalize_address,
)


# ============================================================
# PATHS
# ============================================================

ROOT = os.getcwd()

TRAIN = os.path.join(ROOT, "dataset", "train")
TEST = os.path.join(ROOT, "dataset", "test")

S1_TRAIN = os.path.join(TRAIN, "train_source1.tsv")
S2_TRAIN = os.path.join(TRAIN, "train_source2.tsv")
S3_TRAIN = os.path.join(TRAIN, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN, "train_ground_truth.tsv")

S1_TEST = os.path.join(TEST, "test_source1.tsv")
S2_TEST = os.path.join(TEST, "test_source2.tsv")
S3_TEST = os.path.join(TEST, "test_source3.tsv")

OUTPUT = os.path.join(ROOT, "output")
MODELS = os.path.join(ROOT, "models")

MODEL_FILE = os.path.join(
    MODELS,
    "entity_matcher.joblib",
)

MATCH_FILE = os.path.join(
    OUTPUT,
    "matching_results.tsv",
)

CANDIDATE_FILE = os.path.join(
    OUTPUT,
    "candidate_pairs.tsv",
)


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 100_000

# About 100k S1 records are enough to train a first model
# under the 3-hour deadline.
TRAIN_S1_LIMIT = 100_000

# Candidate counts.
TRAIN_TOP_K = 30
TEST_TOP_K = 15

# Blocking limits.
NAME_BUCKET_LIMIT = 100
ADDRESS_BUCKET_LIMIT = 50
FALLBACK_ADDRESS_LIMIT = 100

ADDRESS_TOP_K = 2

MIN_NAME_TOKEN_LEN = 3
MIN_ADDRESS_TOKEN_LEN = 3

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def read_chunks(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )


def safe_len_ratio(a, b):
    if not a or not b:
        return 0.0

    return min(
        len(a),
        len(b),
    ) / max(
        len(a),
        len(b),
    )


def code_source_index(code):
    """
    Positive code -> Source 2
    Negative code -> Source 3
    """

    if code > 0:
        return 2, code - 1

    return 3, -code - 1


# ============================================================
# BUILD SOURCE INDEX
# ============================================================

def build_source(path, source_number):

    print(
        f"\nBuilding Source {source_number}: "
        f"{os.path.basename(path)}"
    )

    ids = []
    names = []
    addresses = []

    exact_name = defaultdict(list)
    name_tokens = defaultdict(list)

    exact_address = defaultdict(list)
    address_tokens = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_chunks(path),
        start=1,
    ):

        print(
            f"  processed chunk {chunk_no}"
        )

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id
            country = row.country

            raw_name = row.business_name
            raw_address = row.business_address

            name = (
                normalize_name_v2(raw_name)
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
                normalize_address(raw_address)
                if raw_address
                else ""
            )

            row_index = len(ids)

            if source_number == 2:
                code = row_index + 1
            else:
                code = -(row_index + 1)

            ids.append(entity_id)
            names.append(name)
            addresses.append(address)

            # --------------------------------------------
            # Exact/token-sorted name
            # --------------------------------------------

            if sorted_name:

                exact_name[
                    (
                        country,
                        sorted_name,
                    )
                ].append(code)

            # --------------------------------------------
            # Rare name tokens
            # --------------------------------------------

            if name:

                for token in set(
                    token
                    for token in name.split()
                    if len(token)
                    >= MIN_NAME_TOKEN_LEN
                ):

                    name_tokens[
                        (
                            country,
                            token,
                        )
                    ].append(code)

            # --------------------------------------------
            # Exact address
            # --------------------------------------------

            if address:

                exact_address[
                    (
                        country,
                        address,
                    )
                ].append(code)

                # ----------------------------------------
                # Address tokens
                # ----------------------------------------

                for token in set(
                    token
                    for token in address.split()
                    if len(token)
                    >= MIN_ADDRESS_TOKEN_LEN
                ):

                    address_tokens[
                        (
                            country,
                            token,
                        )
                    ].append(code)

    print("Filtering rare name buckets...")

    name_tokens = {
        key: value
        for key, value
        in name_tokens.items()
        if len(value)
        <= NAME_BUCKET_LIMIT
    }

    print(
        f"  retained name buckets: "
        f"{len(name_tokens):,}"
    )

    print("Filtering address buckets...")

    address_tokens = {
        key: value
        for key, value
        in address_tokens.items()
        if len(value)
        <= FALLBACK_ADDRESS_LIMIT
    }

    print(
        f"  retained address buckets: "
        f"{len(address_tokens):,}"
    )

    return {
        "ids": ids,
        "names": names,
        "addresses": addresses,
        "exact_name": dict(exact_name),
        "name_tokens": name_tokens,
        "exact_address": dict(exact_address),
        "address_tokens": address_tokens,
    }


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    country,
    raw_name,
    raw_address,
    s2,
    s3,
    top_k,
):
    """
    Generates scored candidate records.

    Evidence:
        +10 exact token-sorted name
        +10 exact normalized address
        +3 rare name-token hit
        +3 rare address-token hit
        +1 broader fallback address hit
    """

    scores = {}

    def add(code, score):
        scores[code] = (
            scores.get(code, 0)
            + score
        )

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    if raw_name:

        sorted_name = (
            normalize_name_token_sorted(
                raw_name
            )
        )

        key = (
            country,
            sorted_name,
        )

        for code in s2[
            "exact_name"
        ].get(key, []):

            add(code, 10)

        for code in s3[
            "exact_name"
        ].get(key, []):

            add(code, 10)

        normalized_name = (
            normalize_name_v2(
                raw_name
            )
        )

        tokens = {
            token
            for token in normalized_name.split()
            if len(token)
            >= MIN_NAME_TOKEN_LEN
        }

        for token in tokens:

            key = (
                country,
                token,
            )

            for code in s2[
                "name_tokens"
            ].get(key, []):

                add(code, 3)

            for code in s3[
                "name_tokens"
            ].get(key, []):

                add(code, 3)

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    if raw_address:

        address = normalize_address(
            raw_address
        )

        key = (
            country,
            address,
        )

        for code in s2[
            "exact_address"
        ].get(key, []):

            add(code, 10)

        for code in s3[
            "exact_address"
        ].get(key, []):

            add(code, 10)

        tokens = {
            token
            for token in address.split()
            if len(token)
            >= MIN_ADDRESS_TOKEN_LEN
        }

        rarity = []

        for token in tokens:

            key = (
                country,
                token,
            )

            ids2 = s2[
                "address_tokens"
            ].get(key, [])

            ids3 = s3[
                "address_tokens"
            ].get(key, [])

            sizes = []

            if ids2:
                sizes.append(
                    len(ids2)
                )

            if ids3:
                sizes.append(
                    len(ids3)
                )

            if sizes:

                rarity.append(
                    (
                        min(sizes),
                        token,
                    )
                )

        rarity.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        # Strong address tokens <= 50
        for bucket_size, token in rarity:

            if bucket_size > ADDRESS_BUCKET_LIMIT:
                continue

            key = (
                country,
                token,
            )

            for code in s2[
                "address_tokens"
            ].get(key, []):

                add(code, 3)

            for code in s3[
                "address_tokens"
            ].get(key, []):

                add(code, 3)

        # Adaptive fallback.
        if (
            len(scores) <= 20
        ):

            for bucket_size, token in rarity[
                :ADDRESS_TOP_K
            ]:

                if (
                    bucket_size
                    > FALLBACK_ADDRESS_LIMIT
                ):
                    continue

                key = (
                    country,
                    token,
                )

                for code in s2[
                    "address_tokens"
                ].get(key, []):

                    add(code, 1)

                for code in s3[
                    "address_tokens"
                ].get(key, []):

                    add(code, 1)

    if not scores:
        return {}

    ranked = heapq.nlargest(
        top_k,
        scores.items(),
        key=lambda item: (
            item[1],
            -abs(item[0]),
        ),
    )

    return dict(ranked)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

FEATURE_NAMES = [
    "exact_name",
    "exact_address",
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "name_length_ratio",
    "address_length_ratio",
    "block_score",
    "is_source3",
    "target_name_missing",
    "target_address_missing",
]


def make_features(
    s1_name,
    s1_address,
    scored,
    s2,
    s3,
):

    name1 = (
        normalize_name_v2(
            s1_name
        )
        if s1_name
        else ""
    )

    address1 = (
        normalize_address(
            s1_address
        )
        if s1_address
        else ""
    )

    sorted_name1 = (
        normalize_name_token_sorted(
            s1_name
        )
        if s1_name
        else ""
    )

    codes = list(
        scored.keys()
    )

    X = np.zeros(
        (
            len(codes),
            len(FEATURE_NAMES),
        ),
        dtype=np.float32,
    )

    for i, code in enumerate(codes):

        source, index = (
            code_source_index(code)
        )

        target = (
            s2
            if source == 2
            else s3
        )

        name2 = target[
            "names"
        ][index]

        address2 = target[
            "addresses"
        ][index]

        sorted_name2 = (
            normalize_name_token_sorted(
                name2
            )
            if name2
            else ""
        )

        exact_name = (
            bool(
                sorted_name1
                and sorted_name2
                and sorted_name1
                == sorted_name2
            )
        )

        exact_address = (
            bool(
                address1
                and address2
                and address1
                == address2
            )
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

        name_token_ratio = (
            fuzz.token_set_ratio(
                name1,
                name2,
            )
            / 100.0
            if name1
            and name2
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

        address_token_ratio = (
            fuzz.token_set_ratio(
                address1,
                address2,
            )
            / 100.0
            if address1
            and address2
            else 0.0
        )

        X[i] = [
            float(exact_name),
            float(exact_address),
            name_ratio,
            name_token_ratio,
            address_ratio,
            address_token_ratio,
            safe_len_ratio(
                name1,
                name2,
            ),
            safe_len_ratio(
                address1,
                address2,
            ),
            float(
                scored[code]
            ),
            float(
                source == 3
            ),
            float(
                not bool(name2)
            ),
            float(
                not bool(address2)
            ),
        ]

    return codes, X


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(selected_ids):

    print(
        "\nLoading ground truth..."
    )

    result = {}

    for chunk in read_chunks(
        GT_FILE
    ):

        for row in chunk.itertuples(
            index=False
        ):

            sid = (
                row.source1_entity_id
            )

            if sid not in selected_ids:
                continue

            value = (
                row.matched_entity_ids
                .strip()
            )

            if value:

                result[sid] = {
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                }

            else:

                result[sid] = set()

    return result


# ============================================================
# F0.5
# ============================================================

def f05(true_ids, predicted_ids):

    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    if not true_ids:

        return (
            1.0
            if not predicted_ids
            else 0.0
        )

    if not predicted_ids:
        return 0.0

    tp = len(
        true_ids
        & predicted_ids
    )

    if tp == 0:
        return 0.0

    precision = (
        tp
        / len(predicted_ids)
    )

    recall = (
        tp
        / len(true_ids)
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
# TRAIN
# ============================================================

def train_model():

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING ENTITY RESOLUTION MODEL"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Select training S1 records
    # --------------------------------------------------------

    selected_rows = []
    seen = 0

    for chunk in read_chunks(
        S1_TRAIN
    ):

        for row in chunk.itertuples(
            index=False
        ):

            # Approximately uniform sample across the file.
            if seen % 22 == 0:

                selected_rows.append(
                    (
                        row.entity_id,
                        row.country,
                        row.business_name,
                        row.business_address,
                    )
                )

                if (
                    len(selected_rows)
                    >= TRAIN_S1_LIMIT
                ):
                    break

            seen += 1

        if (
            len(selected_rows)
            >= TRAIN_S1_LIMIT
        ):
            break

    print(
        f"Training S1 entities: "
        f"{len(selected_rows):,}"
    )

    selected_ids = {
        row[0]
        for row in selected_rows
    }

    ground_truth = (
        load_ground_truth(
            selected_ids
        )
    )

    # --------------------------------------------------------
    # Build source indexes
    # --------------------------------------------------------

    s2 = build_source(
        S2_TRAIN,
        2,
    )

    s3 = build_source(
        S3_TRAIN,
        3,
    )

    X_batches = []
    y_batches = []

    validation_rows = []

    for position, (
        sid,
        country,
        raw_name,
        raw_address,
    ) in enumerate(
        selected_rows
    ):

        candidates = generate_candidates(
            country,
            raw_name,
            raw_address,
            s2,
            s3,
            TRAIN_TOP_K,
        )

        if (
            position % 10 == 0
        ):

            validation_rows.append(
                (
                    sid,
                    country,
                    raw_name,
                    raw_address,
                    candidates,
                )
            )

        elif candidates:

            codes, X = make_features(
                raw_name,
                raw_address,
                candidates,
                s2,
                s3,
            )

            target_ids = [
                (
                    s2["ids"][code - 1]
                    if code > 0
                    else s3["ids"][-code - 1]
                )
                for code in codes
            ]

            truth = ground_truth.get(
                sid,
                set(),
            )

            y = np.asarray(
                [
                    (
                        1
                        if entity_id in truth
                        else 0
                    )
                    for entity_id in target_ids
                ],
                dtype=np.int8,
            )

            X_batches.append(X)
            y_batches.append(y)

        if (
            position + 1
        ) % 10_000 == 0:

            print(
                f"Prepared "
                f"{position + 1:,} "
                f"training S1 records"
            )

    X_train = np.vstack(
        X_batches
    )

    y_train = np.concatenate(
        y_batches
    )

    print(
        f"\nTraining pairs: "
        f"{len(y_train):,}"
    )

    print(
        f"Positive pairs: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Negative pairs: "
        f"{int((y_train == 0).sum()):,}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    print(
        "\nFitting Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Model training complete."
    )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    print(
        "\nTuning macro F0.5..."
    )

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    best_threshold = 0.70
    best_score = -1.0

    for threshold in thresholds:

        entity_scores = []

        for (
            sid,
            country,
            raw_name,
            raw_address,
            candidates,
        ) in validation_rows:

            if not candidates:

                predicted = []

            else:

                codes, X = make_features(
                    raw_name,
                    raw_address,
                    candidates,
                    s2,
                    s3,
                )

                probabilities = (
                    model.predict_proba(X)[:, 1]
                )

                predicted = []

                for i, (
                    code,
                    probability,
                ) in enumerate(
                    zip(
                        codes,
                        probabilities,
                    )
                ):

                    # Exact normalized/token-sorted
                    # name/address are treated as
                    # deterministic high-confidence evidence.
                    if (
                        X[i, 0] >= 1
                        or X[i, 1] >= 1
                        or probability
                        >= threshold
                    ):

                        entity_id = (
                            s2["ids"][code - 1]
                            if code > 0
                            else s3["ids"][-code - 1]
                        )

                        predicted.append(
                            entity_id
                        )

            score = f05(
                ground_truth.get(
                    sid,
                    set(),
                ),
                predicted,
            )

            entity_scores.append(
                score
            )

        mean_score = (
            float(
                np.mean(
                    entity_scores
                )
            )
            if entity_scores
            else 0.0
        )

        print(
            f"Threshold "
            f"{threshold:.2f}"
            f" -> F0.5 "
            f"{mean_score:.5f}"
        )

        if mean_score > best_score:

            best_score = mean_score
            best_threshold = threshold

    print(
        "\nBEST THRESHOLD: "
        f"{best_threshold:.2f}"
    )

    print(
        "VALIDATION MACRO F0.5: "
        f"{best_score:.5f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        MODELS,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "threshold": float(
                best_threshold
            ),
            "features": FEATURE_NAMES,
            "config": {
                "train_s1": TRAIN_S1_LIMIT,
                "train_top_k": TRAIN_TOP_K,
                "test_top_k": TEST_TOP_K,
                "name_bucket_limit": NAME_BUCKET_LIMIT,
                "address_bucket_limit":
                    ADDRESS_BUCKET_LIMIT,
                "fallback_address_limit":
                    FALLBACK_ADDRESS_LIMIT,
            },
        },
        MODEL_FILE,
    )

    print(
        f"\nSaved model:\n"
        f"{MODEL_FILE}"
    )

    del X_train
    del y_train
    del X_batches
    del y_batches
    del validation_rows
    del ground_truth
    del s2
    del s3

    gc.collect()

    return (
        model,
        float(best_threshold),
    )


# ============================================================
# TEST INFERENCE
# ============================================================

def infer_test(
    model,
    threshold,
):

    print(
        "\n" + "=" * 80
    )

    print(
        "TEST INFERENCE"
    )

    print(
        "=" * 80
    )

    s2 = build_source(
        S2_TEST,
        2,
    )

    s3 = build_source(
        S3_TEST,
        3,
    )

    os.makedirs(
        OUTPUT,
        exist_ok=True,
    )

    with open(
        MATCH_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as matching_fp, open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as candidate_fp:

        matching_writer = csv.writer(
            matching_fp,
            delimiter="\t",
            lineterminator="\n",
        )

        candidate_writer = csv.writer(
            candidate_fp,
            delimiter="\t",
            lineterminator="\n",
        )

        matching_writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        candidate_writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

        total_s1 = 0
        total_candidates = 0
        total_matches = 0

        for chunk_no, chunk in enumerate(
            read_chunks(S1_TEST),
            start=1,
        ):

            for row in chunk.itertuples(
                index=False
            ):

                candidates = generate_candidates(
                    row.country,
                    row.business_name,
                    row.business_address,
                    s2,
                    s3,
                    TEST_TOP_K,
                )

                codes = list(
                    candidates.keys()
                )

                ids = [
                    (
                        s2["ids"][code - 1]
                        if code > 0
                        else s3["ids"][-code - 1]
                    )
                    for code in codes
                ]

                # Stable candidate order.
                pairs = sorted(
                    zip(
                        codes,
                        ids,
                    ),
                    key=lambda x: x[1],
                )

                codes = [
                    x[0]
                    for x in pairs
                ]

                ids = [
                    x[1]
                    for x in pairs
                ]

                # ------------------------------------------------
                # EXACT CANDIDATE SET FED TO MODEL
                # ------------------------------------------------

                candidate_writer.writerow(
                    [
                        row.entity_id,
                        ",".join(ids),
                    ]
                )

                total_candidates += len(
                    ids
                )

                predictions = []

                if codes:

                    ordered_scores = {
                        code: candidates[code]
                        for code in codes
                    }

                    feature_codes, X = (
                        make_features(
                            row.business_name,
                            row.business_address,
                            ordered_scores,
                            s2,
                            s3,
                        )
                    )

                    probabilities = (
                        model.predict_proba(X)[:, 1]
                    )

                    for i, (
                        code,
                        probability,
                    ) in enumerate(
                        zip(
                            feature_codes,
                            probabilities,
                        )
                    ):

                        if (
                            X[i, 0] >= 1
                            or X[i, 1] >= 1
                            or probability
                            >= threshold
                        ):

                            entity_id = (
                                s2["ids"][code - 1]
                                if code > 0
                                else s3["ids"][-code - 1]
                            )

                            predictions.append(
                                entity_id
                            )

                predictions = sorted(
                    set(predictions)
                )

                total_matches += len(
                    predictions
                )

                matching_writer.writerow(
                    [
                        row.entity_id,
                        ",".join(
                            predictions
                        ),
                    ]
                )

                total_s1 += 1

            print(
                f"Processed test chunk "
                f"{chunk_no}"
            )

    print(
        "\nTEST COMPLETE"
    )

    print(
        f"S1 entities: "
        f"{total_s1:,}"
    )

    print(
        f"Candidate relationships: "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{total_candidates / total_s1:.2f}"
    )

    print(
        f"Predicted matches: "
        f"{total_matches:,}"
    )

    print(
        "\nFiles created:"
    )

    print(
        MATCH_FILE
    )

    print(
        CANDIDATE_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "AMAZON ML 2026"
    )

    print(
        "BUSINESS ENTITY RESOLUTION"
    )

    print(
        "PAIRWISE ML PIPELINE"
    )

    print(
        "=" * 80
    )

    model, threshold = train_model()

    infer_test(
        model,
        threshold,
    )

    print(
        "\nALL DONE."
    )


if __name__ == "__main__":
    main()