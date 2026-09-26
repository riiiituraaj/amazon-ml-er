import os
import heapq
from collections import defaultdict

import pandas as pd

from normalization import (
    normalize_name,
    normalize_name_token_sorted,
    normalize_address,
    address_tokens,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

CHUNK_SIZE = 100_000

# Fast experiment size.
# 200,000 S1 rows is enough for a first comparison.
MAX_EVAL_ROWS = 2_206_821

# E016 / E017 base configuration
NAME_BUCKET_LIMIT = 100
ADDRESS_BUCKET_LIMIT = 50
ONE_SIGNAL_LIMIT = 20

# Broader fallback
FALLBACK_ADDRESS_LIMIT = 100
FALLBACK_THRESHOLD = 20

MIN_NAME_TOKEN_LEN = 3
MIN_ADDRESS_TOKEN_LEN = 3

ADDRESS_TOP_K = 2

TOP_K_VALUES = [75]


# ============================================================
# READER
# ============================================================

def read_chunks(path, nrows=None):

    kwargs = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "chunksize": CHUNK_SIZE,
    }

    if nrows is not None:
        kwargs["nrows"] = nrows

    return pd.read_csv(
        path,
        **kwargs,
    )


# ============================================================
# TOKEN-SORTED NAME INDEX
# ============================================================

def build_token_sorted_name_index(path):

    print(
        f"Building token-sorted name index: "
        f"{os.path.basename(path)}"
    )

    index = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_chunks(path),
        start=1,
    ):

        print(f"  processed chunk {chunk_no}")

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = row.country
            name = row.business_name

            if not name:
                continue

            normalized = normalize_name_token_sorted(
                name
            )

            if normalized:
                index[
                    (country, normalized)
                ].append(entity_id)

    print(
        f"  unique keys: {len(index):,}"
    )

    return index


# ============================================================
# NAME TOKEN INDEX
# ============================================================

def build_name_token_index(path):

    print(
        f"Building name token index: "
        f"{os.path.basename(path)}"
    )

    buckets = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_chunks(path),
        start=1,
    ):

        print(f"  processed chunk {chunk_no}")

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = row.country
            name = row.business_name

            if not name:
                continue

            normalized = normalize_name(name)

            tokens = {
                token
                for token in normalized.split()
                if len(token) >= MIN_NAME_TOKEN_LEN
            }

            for token in tokens:

                buckets[
                    (country, token)
                ].append(entity_id)

    return buckets


# ============================================================
# ADDRESS TOKEN INDEX
# ============================================================

def build_address_token_index(path):

    print(
        f"Building address token index: "
        f"{os.path.basename(path)}"
    )

    buckets = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_chunks(path),
        start=1,
    ):

        print(f"  processed chunk {chunk_no}")

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = row.country
            address = row.business_address

            if not address:
                continue

            normalized = normalize_address(
                address
            )

            tokens = {
                token
                for token in address_tokens(
                    normalized
                )
                if len(token) >= MIN_ADDRESS_TOKEN_LEN
            }

            for token in tokens:

                buckets[
                    (country, token)
                ].append(entity_id)

    return buckets


# ============================================================
# EXACT ADDRESS INDEX
# ============================================================

def build_exact_address_index(path):

    print(
        f"Building exact address index: "
        f"{os.path.basename(path)}"
    )

    index = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_chunks(path),
        start=1,
    ):

        print(f"  processed chunk {chunk_no}")

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = row.country
            address = row.business_address

            if not address:
                continue

            normalized = normalize_address(
                address
            )

            if normalized:

                index[
                    (country, normalized)
                ].append(entity_id)

    print(
        f"  unique keys: {len(index):,}"
    )

    return index


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(path):

    print("\nLoading ground truth...")

    gt = {}

    for chunk in read_chunks(path):

        for row in chunk.itertuples(index=False):

            s1_id = row.source1_entity_id
            matched = row.matched_entity_ids

            if matched.strip():

                gt[s1_id] = {
                    x.strip()
                    for x in matched.split(",")
                    if x.strip()
                }

            else:

                gt[s1_id] = set()

    return gt


# ============================================================
# SCORE ONE S1 RECORD
# ============================================================

def score_candidates(
    country,
    name,
    address,
    name_exact_s2,
    name_exact_s3,
    name_token_s2,
    name_token_s3,
    address_token_s2,
    address_token_s3,
    exact_address_s2,
    exact_address_s3,
):
    """
    Return a dictionary:

        candidate_id -> integer evidence score

    Scoring:

        +10 exact token-sorted name
        +10 exact normalized address
        +3 rare name token
        +3 rare address token
        +1 broad address fallback

    IMPORTANT:
    The score is only for ranking.
    The ML model will later use richer features.
    """

    scores = defaultdict(int)

    # --------------------------------------------------------
    # Exact token-sorted name
    # --------------------------------------------------------

    if name:

        key = (
            country,
            normalize_name_token_sorted(name),
        )

        for entity_id in name_exact_s2.get(
            key,
            []
        ):

            scores[entity_id] += 10

        for entity_id in name_exact_s3.get(
            key,
            []
        ):

            scores[entity_id] += 10

    # --------------------------------------------------------
    # Exact address
    # --------------------------------------------------------

    if address:

        normalized_address = normalize_address(
            address
        )

        if normalized_address:

            key = (
                country,
                normalized_address,
            )

            for entity_id in exact_address_s2.get(
                key,
                []
            ):

                scores[entity_id] += 10

            for entity_id in exact_address_s3.get(
                key,
                []
            ):

                scores[entity_id] += 10

    # --------------------------------------------------------
    # Name token candidates
    # --------------------------------------------------------

    if name:

        normalized_name = normalize_name(
            name
        )

        tokens = {
            token
            for token in normalized_name.split()
            if len(token) >= MIN_NAME_TOKEN_LEN
        }

        for token in tokens:

            key = (
                country,
                token,
            )

            ids_s2 = name_token_s2.get(
                key,
                []
            )

            ids_s3 = name_token_s3.get(
                key,
                []
            )

            if (
                0 < len(ids_s2)
                <= NAME_BUCKET_LIMIT
            ):

                for entity_id in ids_s2:
                    scores[entity_id] += 3

            if (
                0 < len(ids_s3)
                <= NAME_BUCKET_LIMIT
            ):

                for entity_id in ids_s3:
                    scores[entity_id] += 3

    # --------------------------------------------------------
    # Address token candidates
    # --------------------------------------------------------

    if address:

        normalized_address = normalize_address(
            address
        )

        tokens = {
            token
            for token in address_tokens(
                normalized_address
            )
            if len(token) >= MIN_ADDRESS_TOKEN_LEN
        }

        token_info = []

        for token in tokens:

            key = (
                country,
                token,
            )

            ids_s2 = address_token_s2.get(
                key,
                []
            )

            ids_s3 = address_token_s3.get(
                key,
                []
            )

            sizes = []

            if ids_s2:
                sizes.append(len(ids_s2))

            if ids_s3:
                sizes.append(len(ids_s3))

            if not sizes:
                continue

            rarity = min(sizes)

            if rarity <= ADDRESS_BUCKET_LIMIT:

                token_info.append(
                    (
                        rarity,
                        token,
                    )
                )

        token_info.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        for _rarity, token in token_info[
            :ADDRESS_TOP_K
        ]:

            key = (
                country,
                token,
            )

            for entity_id in address_token_s2.get(
                key,
                []
            ):

                scores[entity_id] += 3

            for entity_id in address_token_s3.get(
                key,
                []
            ):

                scores[entity_id] += 3

            # Broader fallback gets +1 instead of +3.
            #
            # We only add candidates that were not already
            # represented strongly above.
            #
            # Rather than creating another huge candidate pool,
            # cap this fallback by checking the bucket size.

    return scores


# ============================================================
# BROAD FALLBACK
# ============================================================

def add_broad_fallback_scores(
    scores,
    country,
    address,
    address_token_s2,
    address_token_s3,
):
    """
    Add broad E008-style address evidence.

    Only buckets <= FALLBACK_ADDRESS_LIMIT are considered.
    """

    if not address:
        return

    normalized = normalize_address(
        address
    )

    tokens = {
        token
        for token in address_tokens(
            normalized
        )
        if len(token) >= MIN_ADDRESS_TOKEN_LEN
    }

    token_info = []

    for token in tokens:

        key = (
            country,
            token,
        )

        ids_s2 = address_token_s2.get(
            key,
            []
        )

        ids_s3 = address_token_s3.get(
            key,
            []
        )

        sizes = []

        if ids_s2:
            sizes.append(
                len(ids_s2)
            )

        if ids_s3:
            sizes.append(
                len(ids_s3)
            )

        if not sizes:
            continue

        rarity = min(sizes)

        if rarity <= FALLBACK_ADDRESS_LIMIT:

            token_info.append(
                (
                    rarity,
                    token,
                )
            )

    token_info.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    for _rarity, token in token_info[
        :ADDRESS_TOP_K
    ]:

        key = (
            country,
            token,
        )

        for entity_id in address_token_s2.get(
            key,
            []
        ):

            scores[entity_id] += 1

        for entity_id in address_token_s3.get(
            key,
            []
        ):

            scores[entity_id] += 1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("E018 - OPTIMIZED SIGNAL-RANKED BLOCKING")
    print("=" * 80)

    print(
        f"\nEvaluation rows: "
        f"{MAX_EVAL_ROWS:,}"
    )

    print(
        f"Top-K values: "
        f"{TOP_K_VALUES}"
    )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    print(
        "\n1. Token-sorted name indexes"
    )

    name_exact_s2 = build_token_sorted_name_index(
        S2_FILE
    )

    name_exact_s3 = build_token_sorted_name_index(
        S3_FILE
    )

    print(
        "\n2. Name token indexes"
    )

    name_token_s2 = build_name_token_index(
        S2_FILE
    )

    name_token_s3 = build_name_token_index(
        S3_FILE
    )

    print(
        "\n3. Address token indexes"
    )

    address_token_s2 = build_address_token_index(
        S2_FILE
    )

    address_token_s3 = build_address_token_index(
        S3_FILE
    )

    print(
        "\n4. Exact address indexes"
    )

    exact_address_s2 = build_exact_address_index(
        S2_FILE
    )

    exact_address_s3 = build_exact_address_index(
        S3_FILE
    )

    print(
        "\n5. Ground truth"
    )

    ground_truth = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    stats = {}

    for k in TOP_K_VALUES:

        stats[k] = {
            "recovered": 0,
            "candidates": 0,
            "entities": 0,
            "singletons": 0,
            "singleton_candidates": 0,
        }

    evaluated_rows = 0
    total_true = 0

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\n6. Evaluating..."
    )

    for chunk_no, chunk in enumerate(
        read_chunks(
            S1_FILE,
            nrows=MAX_EVAL_ROWS,
        ),
        start=1,
    ):

        for row in chunk.itertuples(index=False):

            if evaluated_rows >= MAX_EVAL_ROWS:
                break

            s1_id = row.entity_id
            country = row.country
            name = row.business_name
            address = row.business_address

            true_matches = ground_truth.get(
                s1_id,
                set(),
            )

            total_true += len(
                true_matches
            )

            # Base signal scores.
            scores = score_candidates(
                country,
                name,
                address,
                name_exact_s2,
                name_exact_s3,
                name_token_s2,
                name_token_s3,
                address_token_s2,
                address_token_s3,
                exact_address_s2,
                exact_address_s3,
            )

            # Adaptive fallback is applied only to small
            # base candidate sets.
            if len(scores) <= FALLBACK_THRESHOLD:

                add_broad_fallback_scores(
                    scores,
                    country,
                    address,
                    address_token_s2,
                    address_token_s3,
                )

            if scores:

                ranked = heapq.nlargest(
                    max(TOP_K_VALUES),
                    scores.items(),
                    key=lambda x: (
                        x[1],
                        x[0],
                    ),
                )

                ranked_ids = [
                    candidate
                    for candidate, _score
                    in ranked
                ]

            else:

                ranked_ids = []

            # ------------------------------------------------
            # Evaluate every K from the same ranking.
            # ------------------------------------------------

            for k in TOP_K_VALUES:

                selected = set(
                    ranked_ids[:k]
                )

                stat = stats[k]

                stat["candidates"] += len(
                    selected
                )

                stat["recovered"] += len(
                    true_matches
                    & selected
                )

                if selected:

                    stat["entities"] += 1

                if not true_matches:

                    stat["singletons"] += 1

                    if selected:

                        stat[
                            "singleton_candidates"
                        ] += 1

            evaluated_rows += 1

        print(
            f"Evaluated approximately "
            f"{evaluated_rows:,} S1 rows"
        )

        if evaluated_rows >= MAX_EVAL_ROWS:
            break

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "E018 - RESULTS SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRows evaluated: "
        f"{evaluated_rows:,}"
    )

    print(
        "\nTopK | Recall | AvgCandidates | "
        "CandidatePairs | EntityRate | SingletonRate"
    )

    print(
        "-" * 105
    )

    for k in TOP_K_VALUES:

        stat = stats[k]

        recall = (
            stat["recovered"]
            / total_true
            if total_true
            else 0.0
        )

        avg_candidates = (
            stat["candidates"]
            / evaluated_rows
            if evaluated_rows
            else 0.0
        )

        entity_rate = (
            stat["entities"]
            / evaluated_rows
            if evaluated_rows
            else 0.0
        )

        singleton_rate = (
            stat["singleton_candidates"]
            / stat["singletons"]
            if stat["singletons"]
            else 0.0
        )

        print(
            f"{k:4d} | "
            f"{recall:7.2%} | "
            f"{avg_candidates:13.2f} | "
            f"{stat['candidates']:14,.0f} | "
            f"{entity_rate:10.2%} | "
            f"{singleton_rate:13.2%}"
        )

    print(
        "\nE017 reference:"
    )

    print(
        "  Full-data recall: 66.69%"
    )

    print(
        "  Avg candidates: 50.81"
    )

    print(
        "  Estimated pairs: ~112.1M"
    )


if __name__ == "__main__":
    main()