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

NAME_BUCKET_LIMIT = 100
ADDRESS_BUCKET_LIMIT = 50

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
# RAW NAME TOKEN INDEX
# ============================================================

def build_name_token_index(path):

    print(
        f"Building raw name token index: "
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
# RAW ADDRESS TOKEN INDEX
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
# FILTER
# ============================================================

def filter_buckets(buckets, limit):

    filtered = {}

    removed = 0

    for key, ids in buckets.items():

        if len(ids) <= limit:
            filtered[key] = ids
        else:
            removed += 1

    print(
        f"  retained: {len(filtered):,}"
    )

    print(
        f"  removed oversized: {removed:,}"
    )

    return filtered


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
# SIGNAL 1
# TOKEN-SORTED NAME
# ============================================================

def get_name_sorted_candidates(
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
# SIGNAL 2
# RARE NAME TOKEN
# ============================================================

def get_rare_name_candidates(
    country,
    name,
    index_s2,
    index_s3,
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

        key = (country, token)

        candidates.update(
            index_s2.get(key, [])
        )

        candidates.update(
            index_s3.get(key, [])
        )

    return candidates


# ============================================================
# SIGNAL 3
# RAREST ADDRESS TOKEN
# ============================================================

def get_rare_address_candidates(
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

    tokens = {
        token
        for token in address_tokens(
            normalized
        )
        if len(token) >= MIN_ADDRESS_TOKEN_LEN
    }

    token_info = []

    for token in tokens:

        key = (country, token)

        count_s2 = len(
            index_s2.get(key, [])
        )

        count_s3 = len(
            index_s3.get(key, [])
        )

        counts = []

        if count_s2:
            counts.append(count_s2)

        if count_s3:
            counts.append(count_s3)

        if counts:

            token_info.append(
                (
                    min(counts),
                    token,
                )
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
            index_s2.get(key, [])
        )

        candidates.update(
            index_s3.get(key, [])
        )

    return candidates


# ============================================================
# SIGNAL 4
# EXACT ADDRESS
# ============================================================

def get_exact_address_candidates(
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
# BUILD EXACT ADDRESS INDEX
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
# MAIN EVALUATION
# ============================================================

def evaluate():

    print("=" * 80)
    print("E015 - CONSENSUS BLOCKING")
    print("=" * 80)

    print(
        "\nConfiguration:"
    )

    print(
        f"  Name bucket <= {NAME_BUCKET_LIMIT}"
    )

    print(
        f"  Address bucket <= "
        f"{ADDRESS_BUCKET_LIMIT}"
    )

    print(
        "\nCandidate retained when:"
    )

    print(
        "  - exact token-sorted name"
    )

    print(
        "  - OR exact normalized address"
    )

    print(
        "  - OR >= 2 independent blocking signals"
    )

    # --------------------------------------------------------
    # BUILD INDEXES
    # --------------------------------------------------------

    print(
        "\n1. Building token-sorted name indexes"
    )

    name_sorted_s2 = build_token_sorted_name_index(
        S2_FILE
    )

    name_sorted_s3 = build_token_sorted_name_index(
        S3_FILE
    )

    print(
        "\n2. Building rare name indexes"
    )

    raw_name_s2 = build_name_token_index(
        S2_FILE
    )

    raw_name_s3 = build_name_token_index(
        S3_FILE
    )

    rare_name_s2 = filter_buckets(
        raw_name_s2,
        NAME_BUCKET_LIMIT,
    )

    rare_name_s3 = filter_buckets(
        raw_name_s3,
        NAME_BUCKET_LIMIT,
    )

    # Release raw dictionaries.
    del raw_name_s2
    del raw_name_s3

    print(
        "\n3. Building rare address indexes"
    )

    raw_address_s2 = build_address_token_index(
        S2_FILE
    )

    raw_address_s3 = build_address_token_index(
        S3_FILE
    )

    rare_address_s2 = filter_buckets(
        raw_address_s2,
        ADDRESS_BUCKET_LIMIT,
    )

    rare_address_s3 = filter_buckets(
        raw_address_s3,
        ADDRESS_BUCKET_LIMIT,
    )

    del raw_address_s2
    del raw_address_s3

    print(
        "\n4. Building exact address indexes"
    )

    exact_address_s2 = build_exact_address_index(
        S2_FILE
    )

    exact_address_s3 = build_exact_address_index(
        S3_FILE
    )

    print(
        "\n5. Loading ground truth"
    )

    ground_truth = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_s1 = 0
    total_true = 0

    recovered = 0
    total_candidates = 0

    entities_with_candidates = 0

    singleton_total = 0
    singleton_with_candidates = 0

    signal1_recovered = 0
    signal4_recovered = 0
    consensus_recovered = 0

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print(
        "\n6. Evaluating consensus blocker"
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
                set()
            )

            total_s1 += 1
            total_true += len(
                true_matches
            )

            # ------------------------------------------------
            # Generate signals
            # ------------------------------------------------

            c1 = get_name_sorted_candidates(
                country,
                name,
                name_sorted_s2,
                name_sorted_s3,
            )

            c2 = get_rare_name_candidates(
                country,
                name,
                rare_name_s2,
                rare_name_s3,
            )

            c3 = get_rare_address_candidates(
                country,
                address,
                rare_address_s2,
                rare_address_s3,
            )

            c4 = get_exact_address_candidates(
                country,
                address,
                exact_address_s2,
                exact_address_s3,
            )

            signal1_recovered += len(
                true_matches & c1
            )

            signal4_recovered += len(
                true_matches & c4
            )

            # ------------------------------------------------
            # Count independent signals
            # ------------------------------------------------

            counts = defaultdict(int)

            for candidate in c1:
                counts[candidate] += 1

            for candidate in c2:
                counts[candidate] += 1

            for candidate in c3:
                counts[candidate] += 1

            for candidate in c4:
                counts[candidate] += 1

            # ------------------------------------------------
            # Consensus filtering
            # ------------------------------------------------

            final_candidates = set()

            for candidate, signal_count in counts.items():

                # Exact token-sorted name is strong.
                if candidate in c1:
                    final_candidates.add(candidate)
                    continue

                # Exact normalized address is strong.
                if candidate in c4:
                    final_candidates.add(candidate)
                    continue

                # Otherwise require independent agreement.
                if signal_count >= 2:
                    final_candidates.add(candidate)

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            total_candidates += len(
                final_candidates
            )

            if final_candidates:

                entities_with_candidates += 1

            recovered_for_entity = len(
                true_matches
                & final_candidates
            )

            recovered += recovered_for_entity

            consensus_recovered += recovered_for_entity

            if not true_matches:

                singleton_total += 1

                if final_candidates:
                    singleton_with_candidates += 1

        print(
            f"Evaluated S1 chunk {chunk_no}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

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
        avg_candidates * total_s1
    )

    entity_candidate_rate = (
        entities_with_candidates / total_s1
        if total_s1
        else 0.0
    )

    singleton_candidate_rate = (
        singleton_with_candidates
        / singleton_total
        if singleton_total
        else 0.0
    )

    print("\n" + "=" * 80)
    print("E015 - CONSENSUS BLOCKING RESULTS")
    print("=" * 80)

    print(
        f"\nS1 entities: {total_s1:,}"
    )

    print(
        f"True links: {total_true:,}"
    )

    print(
        f"Recovered true links: {recovered:,}"
    )

    print(
        f"\nCandidate Recall: "
        f"{recall:.4%}"
    )

    print(
        f"Average candidates per S1: "
        f"{avg_candidates:.4f}"
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


if __name__ == "__main__":
    evaluate()