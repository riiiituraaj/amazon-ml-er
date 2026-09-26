import os
from collections import defaultdict

import pandas as pd

from normalization import (
    normalize_address,
    address_tokens,
    extract_numbers,
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

MIN_TOKEN_LEN = 3

MAX_BUCKET_SIZE = 100


# ============================================================
# READ CHUNKS
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
# BUILD INDEX
# ============================================================

def build_index(path):

    print(
        f"Building house-number + address-token index: "
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

            if not normalized:
                continue

            # ------------------------------------------------
            # Extract numeric components
            # ------------------------------------------------

            numbers = extract_numbers(normalized)

            if not numbers:
                continue

            # ------------------------------------------------
            # Extract address tokens
            # ------------------------------------------------

            tokens = {
                token
                for token in address_tokens(normalized)
                if len(token) >= MIN_TOKEN_LEN
            }

            if not tokens:
                continue

            # ------------------------------------------------
            # Combined keys
            # ------------------------------------------------

            for number in numbers:

                for token in tokens:

                    key = (
                        country,
                        number,
                        token,
                    )

                    buckets[key].append(
                        entity_id
                    )

    print(
        "\nFiltering common compound keys..."
    )

    index = {}
    removed = 0

    for key, entity_ids in buckets.items():

        if len(entity_ids) <= MAX_BUCKET_SIZE:

            index[key] = entity_ids

        else:

            removed += 1

    print(
        f"Retained compound keys: "
        f"{len(index):,}"
    )

    print(
        f"Removed oversized keys: "
        f"{removed:,}"
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
# CANDIDATES
# ============================================================

def get_candidates(
    country,
    address,
    index_s2,
    index_s3,
):

    if not address:
        return set()

    normalized = normalize_address(
        address
    )

    if not normalized:
        return set()

    numbers = extract_numbers(
        normalized
    )

    if not numbers:
        return set()

    tokens = {
        token
        for token in address_tokens(
            normalized
        )
        if len(token) >= MIN_TOKEN_LEN
    }

    if not tokens:
        return set()

    candidates = set()

    for number in numbers:

        for token in tokens:

            key = (
                country,
                number,
                token,
            )

            candidates.update(
                index_s2.get(
                    key,
                    []
                )
            )

            candidates.update(
                index_s3.get(
                    key,
                    []
                )
            )

    return candidates


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print("=" * 80)
    print("E014 - HOUSE NUMBER + RARE ADDRESS TOKEN BLOCKING")
    print("=" * 80)

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    print(
        "\n1. Building S2 compound index"
    )

    index_s2 = build_index(
        S2_FILE
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    print(
        "\n2. Building S3 compound index"
    )

    index_s3 = build_index(
        S3_FILE
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print(
        "\n3. Loading ground truth"
    )

    ground_truth = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\n4. Evaluating blocker"
    )

    print(
        "Evaluating country + number + address-token blocking..."
    )

    total_s1 = 0
    total_true_links = 0
    recovered_links = 0

    total_candidates = 0
    entities_with_candidates = 0

    singleton_total = 0
    singleton_with_candidates = 0

    for chunk_no, chunk in enumerate(
        read_chunks(S1_FILE),
        start=1,
    ):

        for row in chunk.itertuples(index=False):

            s1_id = row.entity_id
            country = row.country
            address = row.business_address

            true_matches = ground_truth.get(
                s1_id,
                set()
            )

            candidates = get_candidates(
                country,
                address,
                index_s2,
                index_s3,
            )

            candidate_count = len(
                candidates
            )

            total_s1 += 1

            total_true_links += len(
                true_matches
            )

            recovered_links += len(
                true_matches & candidates
            )

            if candidate_count > 0:

                entities_with_candidates += 1
                total_candidates += candidate_count

            if not true_matches:

                singleton_total += 1

                if candidate_count > 0:

                    singleton_with_candidates += 1

        print(
            f"Evaluated S1 chunk {chunk_no}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    recall = (
        recovered_links
        / total_true_links
        if total_true_links
        else 0.0
    )

    average_candidates = (
        total_candidates
        / total_s1
        if total_s1
        else 0.0
    )

    entity_candidate_rate = (
        entities_with_candidates
        / total_s1
        if total_s1
        else 0.0
    )

    singleton_candidate_rate = (
        singleton_with_candidates
        / singleton_total
        if singleton_total
        else 0.0
    )

    estimated_pairs = (
        average_candidates
        * total_s1
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "E014 - HOUSE NUMBER + RARE ADDRESS TOKEN RESULTS"
    )
    print("=" * 80)

    print(
        f"\nS1 entities: "
        f"{total_s1:,}"
    )

    print(
        f"True links: "
        f"{total_true_links:,}"
    )

    print(
        f"Recovered true links: "
        f"{recovered_links:,}"
    )

    print(
        f"\nCandidate Recall: "
        f"{recall:.4%}"
    )

    print(
        f"Average candidates per S1: "
        f"{average_candidates:.4f}"
    )

    print(
        f"Estimated total candidate pairs: "
        f"{estimated_pairs:,.0f}"
    )

    print(
        f"Entities with >=1 candidate: "
        f"{entities_with_candidates:,} "
        f"({entity_candidate_rate:.2%})"
    )

    print(
        f"\nSingletons: "
        f"{singleton_total:,}"
    )

    print(
        f"Singletons receiving candidates: "
        f"{singleton_with_candidates:,} "
        f"({singleton_candidate_rate:.2%})"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate()