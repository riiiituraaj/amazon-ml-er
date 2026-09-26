import os
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

# ----------------------------
# E016 BASE CONFIGURATION
# ----------------------------

NAME_BUCKET_LIMIT = 100
ADDRESS_BUCKET_LIMIT = 50
ONE_SIGNAL_LIMIT = 20

MIN_NAME_TOKEN_LEN = 3
MIN_ADDRESS_TOKEN_LEN = 3

ADDRESS_TOP_K = 2

# ----------------------------
# ADAPTIVE FALLBACK
# ----------------------------

# Apply the broader E008-style address expansion only
# when the base candidate set is at or below one of
# these sizes.
FALLBACK_THRESHOLDS = [0, 5, 10, 20]

# Broader address blocker corresponding to E008.
FALLBACK_ADDRESS_BUCKET_LIMIT = 100


# ============================================================
# READER
# ============================================================

def read_chunks(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
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
# NAME TOKEN BUCKETS
# ============================================================

def build_name_token_buckets(path):

    print(
        f"Building name token buckets: "
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
# ADDRESS TOKEN BUCKETS
# ============================================================

def build_address_token_buckets(path):

    print(
        f"Building address token buckets: "
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
# E016 BASE: EXACT TOKEN-SORTED NAME
# ============================================================

def get_name_exact_candidates(
    country,
    name,
    index_s2,
    index_s3,
):

    candidates = set()

    if not name:
        return candidates

    key = (
        country,
        normalize_name_token_sorted(name),
    )

    candidates.update(
        index_s2.get(key, [])
    )

    candidates.update(
        index_s3.get(key, [])
    )

    return candidates


# ============================================================
# E016 BASE: EXACT ADDRESS
# ============================================================

def get_address_exact_candidates(
    country,
    address,
    index_s2,
    index_s3,
):

    candidates = set()

    if not address:
        return candidates

    normalized = normalize_address(
        address
    )

    if not normalized:
        return candidates

    key = (
        country,
        normalized,
    )

    candidates.update(
        index_s2.get(key, [])
    )

    candidates.update(
        index_s3.get(key, [])
    )

    return candidates


# ============================================================
# E016 BASE: NAME TOKEN SIGNAL
# ============================================================

def get_name_token_candidates(
    country,
    name,
    index_s2,
    index_s3,
    max_bucket_size,
):

    candidates = set()

    if not name:
        return candidates

    normalized = normalize_name(name)

    tokens = {
        token
        for token in normalized.split()
        if len(token) >= MIN_NAME_TOKEN_LEN
    }

    for token in tokens:

        key = (
            country,
            token,
        )

        ids_s2 = index_s2.get(
            key,
            []
        )

        ids_s3 = index_s3.get(
            key,
            []
        )

        if (
            0 < len(ids_s2)
            <= max_bucket_size
        ):
            candidates.update(
                ids_s2
            )

        if (
            0 < len(ids_s3)
            <= max_bucket_size
        ):
            candidates.update(
                ids_s3
            )

    return candidates


# ============================================================
# E016 BASE: ADDRESS TOKEN SIGNAL
# ============================================================

def get_address_token_candidates(
    country,
    address,
    index_s2,
    index_s3,
    max_bucket_size,
    top_k,
):

    candidates = set()

    if not address:
        return candidates

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

        ids_s2 = index_s2.get(
            key,
            []
        )

        ids_s3 = index_s3.get(
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

        token_info.append(
            (
                rarity,
                token,
            )
        )

    token_info.sort(
        key=lambda x: x[0]
    )

    selected_tokens = [
        token
        for rarity, token
        in token_info[:top_k]
        if rarity <= max_bucket_size
    ]

    for token in selected_tokens:

        key = (
            country,
            token,
        )

        ids_s2 = index_s2.get(
            key,
            []
        )

        ids_s3 = index_s3.get(
            key,
            []
        )

        if len(ids_s2) <= max_bucket_size:

            candidates.update(
                ids_s2
            )

        if len(ids_s3) <= max_bucket_size:

            candidates.update(
                ids_s3
            )

    return candidates


# ============================================================
# BUILD BASE E016 CANDIDATES
# ============================================================

def build_base_candidates(
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
):

    # Strong exact signals.
    exact_name = get_name_exact_candidates(
        country,
        name,
        name_sorted_s2,
        name_sorted_s3,
    )

    exact_address = get_address_exact_candidates(
        country,
        address,
        exact_address_s2,
        exact_address_s3,
    )

    # E016 base rare signals.
    rare_name = get_name_token_candidates(
        country,
        name,
        name_tokens_s2,
        name_tokens_s3,
        NAME_BUCKET_LIMIT,
    )

    rare_address = get_address_token_candidates(
        country,
        address,
        address_tokens_s2,
        address_tokens_s3,
        ADDRESS_BUCKET_LIMIT,
        ADDRESS_TOP_K,
    )

    # --------------------------------------------------------
    # Count independent signals.
    # --------------------------------------------------------

    signal_counts = defaultdict(int)

    for entity_id in exact_name:
        signal_counts[entity_id] += 1

    for entity_id in exact_address:
        signal_counts[entity_id] += 1

    for entity_id in rare_name:
        signal_counts[entity_id] += 1

    for entity_id in rare_address:
        signal_counts[entity_id] += 1

    final_candidates = set()

    for entity_id, count in signal_counts.items():

        # Exact normalized/token-sorted name.
        if entity_id in exact_name:
            final_candidates.add(entity_id)
            continue

        # Exact normalized address.
        if entity_id in exact_address:
            final_candidates.add(entity_id)
            continue

        # At least two independent signals.
        if count >= 2:
            final_candidates.add(entity_id)
            continue

    # E016 selective one-signal recovery:
    # permit very rare signals up to ONE_SIGNAL_LIMIT.

    selective_name = get_name_token_candidates(
        country,
        name,
        name_tokens_s2,
        name_tokens_s3,
        ONE_SIGNAL_LIMIT,
    )

    selective_address = get_address_token_candidates(
        country,
        address,
        address_tokens_s2,
        address_tokens_s3,
        ONE_SIGNAL_LIMIT,
        ADDRESS_TOP_K,
    )

    final_candidates.update(
        selective_name
    )

    final_candidates.update(
        selective_address
    )

    return final_candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("E017 - ADAPTIVE BLOCKING")
    print("=" * 80)

    print(
        "\nBase blocker:"
    )

    print(
        f"  Name bucket <= {NAME_BUCKET_LIMIT}"
    )

    print(
        f"  Address bucket <= "
        f"{ADDRESS_BUCKET_LIMIT}"
    )

    print(
        f"  One-signal limit <= "
        f"{ONE_SIGNAL_LIMIT}"
    )

    print(
        "\nFallback:"
    )

    print(
        f"  Broader address bucket <= "
        f"{FALLBACK_ADDRESS_BUCKET_LIMIT}"
    )

    print(
        f"  Fallback thresholds: "
        f"{FALLBACK_THRESHOLDS}"
    )

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------

    print(
        "\n1. Token-sorted name indexes"
    )

    name_sorted_s2 = build_token_sorted_name_index(
        S2_FILE
    )

    name_sorted_s3 = build_token_sorted_name_index(
        S3_FILE
    )

    print(
        "\n2. Name token buckets"
    )

    name_tokens_s2 = build_name_token_buckets(
        S2_FILE
    )

    name_tokens_s3 = build_name_token_buckets(
        S3_FILE
    )

    print(
        "\n3. Address token buckets"
    )

    address_tokens_s2 = build_address_token_buckets(
        S2_FILE
    )

    address_tokens_s3 = build_address_token_buckets(
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
    # STATISTICS
    # --------------------------------------------------------

    stats = {}

    for threshold in FALLBACK_THRESHOLDS:

        stats[threshold] = {
            "recovered": 0,
            "candidates": 0,
            "entities_with_candidates": 0,
            "singleton_total": 0,
            "singleton_with_candidates": 0,
        }

    total_s1 = 0
    total_true = 0

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print(
        "\n6. Evaluating adaptive fallback"
    )

    for chunk_no, chunk in enumerate(
        read_chunks(S1_FILE),
        start=1,
    ):

        for row in chunk.itertuples(index=False):

            s1_id = row.entity_id
            country = row.country
            name = row.business_name
            address = row.business_address

            true_matches = ground_truth.get(
                s1_id,
                set(),
            )

            total_s1 += 1
            total_true += len(
                true_matches
            )

            # ------------------------------------------------
            # Base E016
            # ------------------------------------------------

            base_candidates = build_base_candidates(
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

            base_size = len(
                base_candidates
            )

            # ------------------------------------------------
            # Only compute broader E008-style expansion
            # when the largest configured threshold permits it.
            # ------------------------------------------------

            expanded_address = None

            if (
                FALLBACK_THRESHOLDS
                and base_size
                <= max(FALLBACK_THRESHOLDS)
            ):

                expanded_address = (
                    get_address_token_candidates(
                        country,
                        address,
                        address_tokens_s2,
                        address_tokens_s3,
                        FALLBACK_ADDRESS_BUCKET_LIMIT,
                        ADDRESS_TOP_K,
                    )
                )

            # ------------------------------------------------
            # Evaluate every fallback threshold.
            # ------------------------------------------------

            for threshold in FALLBACK_THRESHOLDS:

                final_candidates = set(
                    base_candidates
                )

                if (
                    base_size <= threshold
                    and expanded_address is not None
                ):

                    final_candidates.update(
                        expanded_address
                    )

                stat = stats[threshold]

                stat["candidates"] += len(
                    final_candidates
                )

                stat["recovered"] += len(
                    true_matches
                    & final_candidates
                )

                if final_candidates:

                    stat[
                        "entities_with_candidates"
                    ] += 1

                if not true_matches:

                    stat[
                        "singleton_total"
                    ] += 1

                    if final_candidates:

                        stat[
                            "singleton_with_candidates"
                        ] += 1

        print(
            f"Evaluated S1 chunk {chunk_no}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("E017 - RESULTS SUMMARY")
    print("=" * 80)

    print(
        "\nFallback | Recall | AvgCandidates | "
        "EstimatedPairs | EntityRate | SingletonRate"
    )

    print(
        "-" * 100
    )

    for threshold in FALLBACK_THRESHOLDS:

        stat = stats[threshold]

        recovered = stat["recovered"]
        candidates = stat["candidates"]

        recall = (
            recovered / total_true
            if total_true
            else 0.0
        )

        avg_candidates = (
            candidates / total_s1
            if total_s1
            else 0.0
        )

        entity_rate = (
            stat["entities_with_candidates"]
            / total_s1
            if total_s1
            else 0.0
        )

        singleton_rate = (
            stat["singleton_with_candidates"]
            / stat["singleton_total"]
            if stat["singleton_total"]
            else 0.0
        )

        print(
            f"{threshold:8d} | "
            f"{recall:7.2%} | "
            f"{avg_candidates:13.2f} | "
            f"{candidates:14,.0f} | "
            f"{entity_rate:10.2%} | "
            f"{singleton_rate:13.2%}"
        )

    print(
        "\nBase E016 reference:"
    )

    print(
        "  Recall: 60.05%"
    )

    print(
        "  Avg candidates: 20.71"
    )

    print(
        "  Candidate pairs: 45,700,683"
    )


if __name__ == "__main__":
    main()