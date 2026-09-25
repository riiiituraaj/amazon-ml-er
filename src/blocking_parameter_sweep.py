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

# We test these limits.
NAME_BUCKET_LIMITS = [300, 150, 100, 50]
ADDRESS_BUCKET_LIMITS = [100, 50, 25]

MIN_NAME_TOKEN_LEN = 3
MIN_ADDRESS_TOKEN_LEN = 3

ADDRESS_TOP_K = 2


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
# BUILD RAW NAME TOKEN BUCKETS
# ============================================================

def build_name_buckets(path):

    print(
        f"Building raw name-token buckets: "
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

    print(
        f"  total raw buckets: "
        f"{len(buckets):,}"
    )

    return buckets


# ============================================================
# BUILD TOKEN-SORTED NAME INDEX
# ============================================================

def build_name_sorted_index(path):

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

            normalized = normalize_name_token_sorted(name)

            if normalized:

                index[
                    (country, normalized)
                ].append(entity_id)

    print(
        f"  unique keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# BUILD RAW ADDRESS TOKEN BUCKETS
# ============================================================

def build_address_buckets(path):

    print(
        f"Building raw address-token buckets: "
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

            normalized = normalize_address(address)

            tokens = {
                token
                for token in address_tokens(normalized)
                if len(token) >= MIN_ADDRESS_TOKEN_LEN
            }

            for token in tokens:

                buckets[
                    (country, token)
                ].append(entity_id)

    print(
        f"  total raw buckets: "
        f"{len(buckets):,}"
    )

    return buckets


# ============================================================
# FILTER BUCKETS
# ============================================================

def filter_buckets(raw_buckets, limit):

    return {
        key: ids
        for key, ids in raw_buckets.items()
        if len(ids) <= limit
    }


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
# CANDIDATES
# ============================================================

def generate_candidates(
    country,
    name,
    address,
    name_sorted_s2,
    name_sorted_s3,
    rare_name_s2,
    rare_name_s3,
    rare_address_s2,
    rare_address_s3,
):

    candidates = set()

    # --------------------------------------------------------
    # Exact token-sorted name
    # --------------------------------------------------------

    if name:

        name_key = (
            country,
            normalize_name_token_sorted(name),
        )

        candidates.update(
            name_sorted_s2.get(
                name_key,
                []
            )
        )

        candidates.update(
            name_sorted_s3.get(
                name_key,
                []
            )
        )

    # --------------------------------------------------------
    # Rare name token
    # --------------------------------------------------------

    if name:

        normalized_name = normalize_name(name)

        name_tokens = {
            token
            for token in normalized_name.split()
            if len(token) >= MIN_NAME_TOKEN_LEN
        }

        for token in name_tokens:

            key = (country, token)

            candidates.update(
                rare_name_s2.get(
                    key,
                    []
                )
            )

            candidates.update(
                rare_name_s3.get(
                    key,
                    []
                )
            )

    # --------------------------------------------------------
    # Rarest address tokens
    # --------------------------------------------------------

    if address:

        normalized_address = normalize_address(
            address
        )

        addr_tokens = {
            token
            for token in address_tokens(
                normalized_address
            )
            if len(token) >= MIN_ADDRESS_TOKEN_LEN
        }

        token_info = []

        for token in addr_tokens:

            key = (country, token)

            count_s2 = len(
                rare_address_s2.get(
                    key,
                    []
                )
            )

            count_s3 = len(
                rare_address_s3.get(
                    key,
                    []
                )
            )

            counts = []

            if count_s2:
                counts.append(count_s2)

            if count_s3:
                counts.append(count_s3)

            if counts:

                rarity = min(counts)

                token_info.append(
                    (rarity, token)
                )

        token_info.sort(
            key=lambda x: x[0]
        )

        selected = [
            token
            for _, token
            in token_info[:ADDRESS_TOP_K]
        ]

        for token in selected:

            key = (country, token)

            candidates.update(
                rare_address_s2.get(
                    key,
                    []
                )
            )

            candidates.update(
                rare_address_s3.get(
                    key,
                    []
                )
            )

    return candidates


# ============================================================
# EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    name_limit,
    address_limit,
    s1_file,
    gt,
    name_sorted_s2,
    name_sorted_s3,
    raw_name_s2,
    raw_name_s3,
    raw_address_s2,
    raw_address_s3,
):

    rare_name_s2 = filter_buckets(
        raw_name_s2,
        name_limit,
    )

    rare_name_s3 = filter_buckets(
        raw_name_s3,
        name_limit,
    )

    rare_address_s2 = filter_buckets(
        raw_address_s2,
        address_limit,
    )

    rare_address_s3 = filter_buckets(
        raw_address_s3,
        address_limit,
    )

    total_s1 = 0
    total_true = 0
    recovered = 0

    total_candidates = 0

    for chunk in read_chunks(s1_file):

        for row in chunk.itertuples(index=False):

            s1_id = row.entity_id
            country = row.country
            name = row.business_name
            address = row.business_address

            true_matches = gt.get(
                s1_id,
                set()
            )

            candidates = generate_candidates(
                country,
                name,
                address,
                name_sorted_s2,
                name_sorted_s3,
                rare_name_s2,
                rare_name_s3,
                rare_address_s2,
                rare_address_s3,
            )

            total_s1 += 1
            total_true += len(
                true_matches
            )

            recovered += len(
                true_matches & candidates
            )

            total_candidates += len(
                candidates
            )

    recall = (
        recovered / total_true
        if total_true
        else 0
    )

    avg_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0
    )

    estimated_pairs = (
        avg_candidates * total_s1
    )

    return (
        recall,
        avg_candidates,
        estimated_pairs,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("E013 - BLOCKING PARAMETER SWEEP")
    print("=" * 80)

    # --------------------------------------------------------
    # BUILD INDEXES ONCE
    # --------------------------------------------------------

    print("\n1. Building token-sorted name indexes")

    name_sorted_s2 = build_name_sorted_index(
        S2_FILE
    )

    name_sorted_s3 = build_name_sorted_index(
        S3_FILE
    )

    print("\n2. Building raw rare-name buckets")

    raw_name_s2 = build_name_buckets(
        S2_FILE
    )

    raw_name_s3 = build_name_buckets(
        S3_FILE
    )

    print("\n3. Building raw address-token buckets")

    raw_address_s2 = build_address_buckets(
        S2_FILE
    )

    raw_address_s3 = build_address_buckets(
        S3_FILE
    )

    print("\n4. Loading ground truth")

    gt = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # SWEEP
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING PARAMETER SWEEP")
    print("=" * 80)

    results = []

    for name_limit in NAME_BUCKET_LIMITS:

        for address_limit in ADDRESS_BUCKET_LIMITS:

            print(
                "\n----------------------------------------"
            )

            print(
                f"Name bucket <= {name_limit}"
            )

            print(
                f"Address bucket <= {address_limit}"
            )

            recall, avg_candidates, estimated_pairs = (
                evaluate_configuration(
                    name_limit,
                    address_limit,
                    S1_FILE,
                    gt,
                    name_sorted_s2,
                    name_sorted_s3,
                    raw_name_s2,
                    raw_name_s3,
                    raw_address_s2,
                    raw_address_s3,
                )
            )

            results.append(
                (
                    name_limit,
                    address_limit,
                    recall,
                    avg_candidates,
                    estimated_pairs,
                )
            )

            print(
                f"Recall: "
                f"{recall:.4%}"
            )

            print(
                f"Average candidates: "
                f"{avg_candidates:.4f}"
            )

            print(
                f"Estimated pairs: "
                f"{estimated_pairs:,.0f}"
            )

    # --------------------------------------------------------
    # SORT BY RECALL
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("E013 - RESULTS SUMMARY")
    print("=" * 80)

    results.sort(
        key=lambda x: (
            -x[2],
            x[3],
        )
    )

    print(
        "\nNameLimit | AddrLimit | Recall | "
        "AvgCandidates | EstimatedPairs"
    )

    print("-" * 80)

    for (
        name_limit,
        address_limit,
        recall,
        avg_candidates,
        estimated_pairs,
    ) in results:

        print(
            f"{name_limit:9d} | "
            f"{address_limit:9d} | "
            f"{recall:7.2%} | "
            f"{avg_candidates:13.2f} | "
            f"{estimated_pairs:15,.0f}"
        )


if __name__ == "__main__":
    main()