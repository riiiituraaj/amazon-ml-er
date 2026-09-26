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

# Base E015 limits.
NAME_BUCKET_LIMIT = 100
ADDRESS_BUCKET_LIMIT = 50

MIN_NAME_TOKEN_LEN = 3
MIN_ADDRESS_TOKEN_LEN = 3

ADDRESS_TOP_K = 2

# We will test these one-signal rarity thresholds.
ONE_SIGNAL_THRESHOLDS = [5, 10, 20]


# ============================================================
# CHUNK READER
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
# EXACT ADDRESS
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
# BASE SIGNALS
# ============================================================

def get_name_sorted_candidates(
    country,
    name,
    index_s2,
    index_s3,
):

    result = set()

    if not name:
        return result

    key = (
        country,
        normalize_name_token_sorted(name),
    )

    result.update(
        index_s2.get(key, [])
    )

    result.update(
        index_s3.get(key, [])
    )

    return result


def get_exact_address_candidates(
    country,
    address,
    index_s2,
    index_s3,
):

    result = set()

    if not address:
        return result

    normalized = normalize_address(
        address
    )

    if not normalized:
        return result

    key = (
        country,
        normalized,
    )

    result.update(
        index_s2.get(key, [])
    )

    result.update(
        index_s3.get(key, [])
    )

    return result


# ============================================================
# RARE NAME SIGNAL
# ============================================================

def get_name_candidates_with_rarity(
    country,
    name,
    buckets_s2,
    buckets_s3,
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

        ids_s2 = buckets_s2.get(
            key,
            []
        )

        ids_s3 = buckets_s3.get(
            key,
            []
        )

        if 0 < len(ids_s2) <= max_bucket_size:

            candidates.update(
                ids_s2
            )

        if 0 < len(ids_s3) <= max_bucket_size:

            candidates.update(
                ids_s3
            )

    return candidates


# ============================================================
# RARE ADDRESS SIGNAL
# ============================================================

def get_address_candidates_with_rarity(
    country,
    address,
    buckets_s2,
    buckets_s3,
    max_bucket_size,
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

        ids_s2 = buckets_s2.get(
            key,
            []
        )

        ids_s3 = buckets_s3.get(
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

        if sizes:

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
        in token_info[:ADDRESS_TOP_K]
        if rarity <= max_bucket_size
    ]

    for token in selected_tokens:

        key = (
            country,
            token,
        )

        ids_s2 = buckets_s2.get(
            key,
            []
        )

        ids_s3 = buckets_s3.get(
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
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_threshold(
    threshold,
    gt,
    name_sorted_s2,
    name_sorted_s3,
    name_buckets_s2,
    name_buckets_s3,
    address_buckets_s2,
    address_buckets_s3,
    exact_address_s2,
    exact_address_s3,
):

    total_s1 = 0
    total_true = 0
    recovered = 0

    total_candidates = 0

    entities_with_candidates = 0

    singleton_total = 0
    singleton_with_candidates = 0

    for chunk in read_chunks(S1_FILE):

        for row in chunk.itertuples(index=False):

            s1_id = row.entity_id
            country = row.country
            name = row.business_name
            address = row.business_address

            true_matches = gt.get(
                s1_id,
                set()
            )

            total_s1 += 1
            total_true += len(
                true_matches
            )

            # ------------------------------------------------
            # Strong exact signals
            # ------------------------------------------------

            c_name_exact = (
                get_name_sorted_candidates(
                    country,
                    name,
                    name_sorted_s2,
                    name_sorted_s3,
                )
            )

            c_address_exact = (
                get_exact_address_candidates(
                    country,
                    address,
                    exact_address_s2,
                    exact_address_s3,
                )
            )

            # ------------------------------------------------
            # Base rare signals
            # ------------------------------------------------

            c_name_rare = (
                get_name_candidates_with_rarity(
                    country,
                    name,
                    name_buckets_s2,
                    name_buckets_s3,
                    NAME_BUCKET_LIMIT,
                )
            )

            c_address_rare = (
                get_address_candidates_with_rarity(
                    country,
                    address,
                    address_buckets_s2,
                    address_buckets_s3,
                    ADDRESS_BUCKET_LIMIT,
                )
            )

            # ------------------------------------------------
            # Count signals
            # ------------------------------------------------

            signal_counts = defaultdict(int)

            for entity_id in c_name_exact:
                signal_counts[
                    entity_id
                ] += 1

            for entity_id in c_address_exact:
                signal_counts[
                    entity_id
                ] += 1

            for entity_id in c_name_rare:
                signal_counts[
                    entity_id
                ] += 1

            for entity_id in c_address_rare:
                signal_counts[
                    entity_id
                ] += 1

            # ------------------------------------------------
            # Selective one-signal recovery
            # ------------------------------------------------

            c_name_selective = (
                get_name_candidates_with_rarity(
                    country,
                    name,
                    name_buckets_s2,
                    name_buckets_s3,
                    threshold,
                )
            )

            c_address_selective = (
                get_address_candidates_with_rarity(
                    country,
                    address,
                    address_buckets_s2,
                    address_buckets_s3,
                    threshold,
                )
            )

            final_candidates = set()

            for entity_id, count in signal_counts.items():

                # Exact name is always retained.
                if entity_id in c_name_exact:

                    final_candidates.add(
                        entity_id
                    )
                    continue

                # Exact address is always retained.
                if entity_id in c_address_exact:

                    final_candidates.add(
                        entity_id
                    )
                    continue

                # Multiple independent signals.
                if count >= 2:

                    final_candidates.add(
                        entity_id
                    )
                    continue

            # One exceptionally rare signal.
            final_candidates.update(
                c_name_selective
            )

            final_candidates.update(
                c_address_selective
            )

            total_candidates += len(
                final_candidates
            )

            recovered += len(
                true_matches
                & final_candidates
            )

            if final_candidates:

                entities_with_candidates += 1

            if not true_matches:

                singleton_total += 1

                if final_candidates:

                    singleton_with_candidates += 1

    recall = (
        recovered / total_true
        if total_true
        else 0.0
    )

    avg_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0.0
    )

    estimated_pairs = (
        total_candidates
    )

    entity_candidate_rate = (
        entities_with_candidates / total_s1
        if total_s1
        else 0.0
    )

    singleton_rate = (
        singleton_with_candidates
        / singleton_total
        if singleton_total
        else 0.0
    )

    return (
        recall,
        avg_candidates,
        estimated_pairs,
        entity_candidate_rate,
        singleton_rate,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("E016 - SELECTIVE ONE-SIGNAL RECOVERY")
    print("=" * 80)

    print(
        "\nBase configuration:"
    )

    print(
        f"  Name bucket <= {NAME_BUCKET_LIMIT}"
    )

    print(
        f"  Address bucket <= "
        f"{ADDRESS_BUCKET_LIMIT}"
    )

    print(
        "\nTesting one-signal rarity thresholds:"
    )

    print(
        f"  {ONE_SIGNAL_THRESHOLDS}"
    )

    # --------------------------------------------------------
    # BUILD INDEXES
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

    name_buckets_s2 = build_name_token_buckets(
        S2_FILE
    )

    name_buckets_s3 = build_name_token_buckets(
        S3_FILE
    )

    print(
        "\n3. Address token buckets"
    )

    address_buckets_s2 = build_address_token_buckets(
        S2_FILE
    )

    address_buckets_s3 = build_address_token_buckets(
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

    gt = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # SWEEP
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "RUNNING E016 SWEEP"
    )

    print(
        "=" * 80
    )

    results = []

    for threshold in ONE_SIGNAL_THRESHOLDS:

        print(
            "\n----------------------------------------"
        )

        print(
            f"One-signal rarity threshold <= "
            f"{threshold}"
        )

        result = evaluate_threshold(
            threshold,
            gt,
            name_sorted_s2,
            name_sorted_s3,
            name_buckets_s2,
            name_buckets_s3,
            address_buckets_s2,
            address_buckets_s3,
            exact_address_s2,
            exact_address_s3,
        )

        (
            recall,
            avg_candidates,
            estimated_pairs,
            entity_rate,
            singleton_rate,
        ) = result

        results.append(
            (
                threshold,
                recall,
                avg_candidates,
                estimated_pairs,
                entity_rate,
                singleton_rate,
            )
        )

        print(
            f"Recall: {recall:.4%}"
        )

        print(
            f"Average candidates: "
            f"{avg_candidates:.4f}"
        )

        print(
            f"Estimated pairs: "
            f"{estimated_pairs:,.0f}"
        )

        print(
            f"Entities with candidates: "
            f"{entity_rate:.2%}"
        )

        print(
            f"Singleton candidate rate: "
            f"{singleton_rate:.2%}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "E016 - RESULTS SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "\nThreshold | Recall | "
        "AvgCandidates | EstimatedPairs | "
        "EntityRate | SingletonRate"
    )

    print(
        "-" * 100
    )

    for (
        threshold,
        recall,
        avg_candidates,
        estimated_pairs,
        entity_rate,
        singleton_rate,
    ) in results:

        print(
            f"{threshold:9d} | "
            f"{recall:7.2%} | "
            f"{avg_candidates:13.2f} | "
            f"{estimated_pairs:15,.0f} | "
            f"{entity_rate:10.2%} | "
            f"{singleton_rate:13.2%}"
        )


if __name__ == "__main__":
    main()