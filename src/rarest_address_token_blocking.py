import os
from collections import defaultdict

import pandas as pd

from normalization import normalize_address, address_tokens


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

# Tokens appearing more than this many times are not useful
# as blocking keys.
MAX_BUCKET_SIZE = 100

# Number of rarest address tokens to use from each S1 address.
TOP_K_TOKENS = 2


# ============================================================
# BUILD INDEX
# ============================================================

def build_address_token_index(path):
    """
    Build:

        (country, token) -> list(entity_id)

    """

    print(f"Building address token index: {os.path.basename(path)}")

    buckets = defaultdict(list)

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
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
                if len(token) >= MIN_TOKEN_LEN
            }

            for token in tokens:
                buckets[(country, token)].append(entity_id)

    print("\nFiltering common address tokens...")

    filtered = {}
    removed = 0

    for key, entity_ids in buckets.items():

        if len(entity_ids) <= MAX_BUCKET_SIZE:
            filtered[key] = entity_ids
        else:
            removed += 1

    print(f"Retained token buckets: {len(filtered):,}")
    print(f"Removed common token buckets: {removed:,}")

    return filtered


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(path):

    print("\nLoading ground truth...")

    gt = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(index=False):

            s1_id = row.source1_entity_id
            matched = row.matched_entity_ids

            if matched.strip():

                ids = {
                    x.strip()
                    for x in matched.split(",")
                    if x.strip()
                }

            else:
                ids = set()

            gt[s1_id] = ids

    return gt


# ============================================================
# GET CANDIDATES
# ============================================================

def get_candidates(
    country,
    address,
    index_s2,
    index_s3,
):
    """
    Use only the TOP_K rarest address tokens from the
    S1 address.

    Rarity is estimated from the bucket size:
    smaller bucket = more discriminative token.
    """

    if not address:
        return set()

    normalized = normalize_address(address)

    tokens = {
        token
        for token in address_tokens(normalized)
        if len(token) >= MIN_TOKEN_LEN
    }

    if not tokens:
        return set()

    token_info = []

    for token in tokens:

        key = (country, token)

        count_s2 = len(index_s2.get(key, []))
        count_s3 = len(index_s3.get(key, []))

        counts = []

        if count_s2 > 0:
            counts.append(count_s2)

        if count_s3 > 0:
            counts.append(count_s3)

        if not counts:
            continue

        # Use the smaller source bucket as an approximation
        # of token discriminativeness.
        rarity = min(counts)

        token_info.append((rarity, token))

    # Smallest bucket first = rarest token first.
    token_info.sort(key=lambda x: x[0])

    selected_tokens = [
        token
        for _, token in token_info[:TOP_K_TOKENS]
    ]

    candidates = set()

    for token in selected_tokens:

        key = (country, token)

        for entity_id in index_s2.get(key, []):
            candidates.add(entity_id)

        for entity_id in index_s3.get(key, []):
            candidates.add(entity_id)

    return candidates


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print("=" * 80)
    print("E010 - RAREST ADDRESS TOKEN BLOCKING")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. S2
    # --------------------------------------------------------

    print("\n1. Building S2 address token index")

    index_s2 = build_address_token_index(S2_FILE)

    # --------------------------------------------------------
    # 2. S3
    # --------------------------------------------------------

    print("\n2. Building S3 address token index")

    index_s3 = build_address_token_index(S3_FILE)

    # --------------------------------------------------------
    # 3. Ground truth
    # --------------------------------------------------------

    print("\n3. Loading ground truth")

    ground_truth = load_ground_truth(GT_FILE)

    # --------------------------------------------------------
    # 4. Evaluate
    # --------------------------------------------------------

    print("\n4. Evaluating blocker")
    print(
        f"Using the {TOP_K_TOKENS} rarest address tokens "
        f"per S1 entity..."
    )

    total_s1 = 0
    total_true_links = 0
    recovered_links = 0

    total_candidates = 0
    entities_with_candidates = 0

    singleton_total = 0
    singleton_with_candidates = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            S1_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
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

            candidate_count = len(candidates)

            total_s1 += 1
            total_true_links += len(true_matches)

            if candidate_count > 0:

                entities_with_candidates += 1
                total_candidates += candidate_count

            recovered_links += len(
                true_matches & candidates
            )

            if len(true_matches) == 0:

                singleton_total += 1

                if candidate_count > 0:
                    singleton_with_candidates += 1

        print(
            f"Evaluated S1 chunk {chunk_no}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    candidate_recall = (
        recovered_links / total_true_links
        if total_true_links
        else 0.0
    )

    average_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0.0
    )

    entity_candidate_rate = (
        entities_with_candidates / total_s1
        if total_s1
        else 0.0
    )

    singleton_candidate_rate = (
        singleton_with_candidates / singleton_total
        if singleton_total
        else 0.0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("E010 - RAREST ADDRESS TOKEN BLOCKING RESULTS")
    print("=" * 80)

    print(
        f"\nS1 entities: {total_s1:,}"
    )

    print(
        f"True links: {total_true_links:,}"
    )

    print(
        f"Recovered true links: {recovered_links:,}"
    )

    print(
        f"\nCandidate Recall: {candidate_recall:.4%}"
    )

    print(
        f"Average candidates per S1: "
        f"{average_candidates:.4f}"
    )

    print(
        f"Entities with >=1 candidate: "
        f"{entities_with_candidates:,} "
        f"({entity_candidate_rate:.2%})"
    )

    print(
        f"\nSingletons: {singleton_total:,}"
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