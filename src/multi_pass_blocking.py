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

# E007
RARE_NAME_MIN_TOKEN_LEN = 3
RARE_NAME_MAX_BUCKET_SIZE = 300

# E010
ADDRESS_MIN_TOKEN_LEN = 3
ADDRESS_MAX_BUCKET_SIZE = 100
ADDRESS_TOP_K = 2


# ============================================================
# GENERIC CHUNK READER
# ============================================================

def read_source_chunks(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )


# ============================================================
# E004 - TOKEN SORTED NAME INDEX
# ============================================================

def build_token_sorted_name_index(path):

    print(
        f"Building token-sorted name index: "
        f"{os.path.basename(path)}"
    )

    index = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_source_chunks(path),
        start=1,
    ):

        print(f"  processed chunk {chunk_no}")

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = row.country
            name = row.business_name

            if not name:
                continue

            key = (
                country,
                normalize_name_token_sorted(name),
            )

            if key[1]:
                index[key].append(entity_id)

    print(
        f"  unique token-sorted name keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# E007 - RARE NAME TOKEN INDEX
# ============================================================

def build_rare_name_token_index(path):

    print(
        f"Building rare name-token index: "
        f"{os.path.basename(path)}"
    )

    buckets = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_source_chunks(path),
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
                if len(token) >= RARE_NAME_MIN_TOKEN_LEN
            }

            for token in tokens:
                buckets[(country, token)].append(entity_id)

    print("\nFiltering common name tokens...")

    index = {}
    removed = 0

    for key, entity_ids in buckets.items():

        if len(entity_ids) <= RARE_NAME_MAX_BUCKET_SIZE:
            index[key] = entity_ids
        else:
            removed += 1

    print(
        f"  retained token buckets: "
        f"{len(index):,}"
    )

    print(
        f"  removed common buckets: "
        f"{removed:,}"
    )

    return index


# ============================================================
# E003 - EXACT NORMALIZED ADDRESS INDEX
# ============================================================

def build_exact_address_index(path):

    print(
        f"Building exact address index: "
        f"{os.path.basename(path)}"
    )

    index = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_source_chunks(path),
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

            index[(country, normalized)].append(entity_id)

    print(
        f"  unique exact-address keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# E010 - RAREST ADDRESS TOKEN INDEX
# ============================================================

def build_address_token_index(path):

    print(
        f"Building rare address-token index: "
        f"{os.path.basename(path)}"
    )

    buckets = defaultdict(list)

    for chunk_no, chunk in enumerate(
        read_source_chunks(path),
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
                if len(token) >= ADDRESS_MIN_TOKEN_LEN
            }

            for token in tokens:
                buckets[(country, token)].append(entity_id)

    print("\nFiltering common address tokens...")

    index = {}
    removed = 0

    for key, entity_ids in buckets.items():

        if len(entity_ids) <= ADDRESS_MAX_BUCKET_SIZE:
            index[key] = entity_ids
        else:
            removed += 1

    print(
        f"  retained token buckets: "
        f"{len(index):,}"
    )

    print(
        f"  removed common buckets: "
        f"{removed:,}"
    )

    return index


# ============================================================
# GROUND TRUTH
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

                gt[s1_id] = {
                    x.strip()
                    for x in matched.split(",")
                    if x.strip()
                }

            else:

                gt[s1_id] = set()

    return gt


# ============================================================
# CANDIDATE GENERATORS
# ============================================================

def token_sorted_name_candidates(
    country,
    name,
    index_s2,
    index_s3,
):

    if not name:
        return set()

    key = (
        country,
        normalize_name_token_sorted(name),
    )

    candidates = set()

    candidates.update(index_s2.get(key, []))
    candidates.update(index_s3.get(key, []))

    return candidates


def rare_name_candidates(
    country,
    name,
    index_s2,
    index_s3,
):

    if not name:
        return set()

    normalized = normalize_name(name)

    tokens = {
        token
        for token in normalized.split()
        if len(token) >= RARE_NAME_MIN_TOKEN_LEN
    }

    candidates = set()

    for token in tokens:

        key = (country, token)

        candidates.update(index_s2.get(key, []))
        candidates.update(index_s3.get(key, []))

    return candidates


def exact_address_candidates(
    country,
    address,
    index_s2,
    index_s3,
):

    if not address:
        return set()

    normalized = normalize_address(address)

    if not normalized:
        return set()

    key = (country, normalized)

    candidates = set()

    candidates.update(index_s2.get(key, []))
    candidates.update(index_s3.get(key, []))

    return candidates


def rarest_address_candidates(
    country,
    address,
    index_s2,
    index_s3,
):

    if not address:
        return set()

    normalized = normalize_address(address)

    tokens = {
        token
        for token in address_tokens(normalized)
        if len(token) >= ADDRESS_MIN_TOKEN_LEN
    }

    if not tokens:
        return set()

    token_info = []

    for token in tokens:

        key = (country, token)

        count_s2 = len(index_s2.get(key, []))
        count_s3 = len(index_s3.get(key, []))

        counts = []

        if count_s2:
            counts.append(count_s2)

        if count_s3:
            counts.append(count_s3)

        if not counts:
            continue

        # Same rarity logic as E010.
        rarity = min(counts)

        token_info.append(
            (rarity, token)
        )

    token_info.sort(
        key=lambda x: x[0]
    )

    selected_tokens = [
        token
        for _, token in token_info[:ADDRESS_TOP_K]
    ]

    candidates = set()

    for token in selected_tokens:

        key = (country, token)

        candidates.update(
            index_s2.get(key, [])
        )

        candidates.update(
            index_s3.get(key, [])
        )

    return candidates


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print("=" * 80)
    print("E011 - MULTI-PASS BLOCKING")
    print("=" * 80)

    print(
        "\nPasses:"
    )

    print(
        "  E004 - Token-sorted name"
    )

    print(
        "  E007 - Rare name token"
    )

    print(
        "  E010 - Rarest address token"
    )

    print(
        "  E003 - Exact normalized address"
    )

    # --------------------------------------------------------
    # BUILD INDEXES
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("1. BUILDING TOKEN-SORTED NAME INDEXES")
    print("-" * 80)

    name_sorted_s2 = build_token_sorted_name_index(
        S2_FILE
    )

    name_sorted_s3 = build_token_sorted_name_index(
        S3_FILE
    )

    print("\n" + "-" * 80)
    print("2. BUILDING RARE NAME TOKEN INDEXES")
    print("-" * 80)

    rare_name_s2 = build_rare_name_token_index(
        S2_FILE
    )

    rare_name_s3 = build_rare_name_token_index(
        S3_FILE
    )

    print("\n" + "-" * 80)
    print("3. BUILDING EXACT ADDRESS INDEXES")
    print("-" * 80)

    exact_address_s2 = build_exact_address_index(
        S2_FILE
    )

    exact_address_s3 = build_exact_address_index(
        S3_FILE
    )

    print("\n" + "-" * 80)
    print("4. BUILDING RAREST ADDRESS TOKEN INDEXES")
    print("-" * 80)

    rare_address_s2 = build_address_token_index(
        S2_FILE
    )

    rare_address_s3 = build_address_token_index(
        S3_FILE
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("5. LOADING GROUND TRUTH")
    print("-" * 80)

    ground_truth = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_s1 = 0
    total_true_links = 0

    recovered_after_name_sorted = 0
    recovered_after_rare_name = 0
    recovered_after_rare_address = 0
    recovered_after_exact_address = 0

    total_final_candidates = 0
    entities_with_candidates = 0

    singleton_total = 0
    singleton_with_candidates = 0

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. EVALUATING UNION")
    print("-" * 80)

    for chunk_no, chunk in enumerate(
        read_source_chunks(S1_FILE),
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
            total_true_links += len(
                true_matches
            )

            # ------------------------------------------------
            # PASS 1 - E004
            # ------------------------------------------------

            c1 = token_sorted_name_candidates(
                country,
                name,
                name_sorted_s2,
                name_sorted_s3,
            )

            union = set(c1)

            recovered_after_name_sorted += len(
                true_matches & union
            )

            # ------------------------------------------------
            # PASS 2 - E007
            # ------------------------------------------------

            c2 = rare_name_candidates(
                country,
                name,
                rare_name_s2,
                rare_name_s3,
            )

            union.update(c2)

            recovered_after_rare_name += len(
                true_matches & union
            )

            # ------------------------------------------------
            # PASS 3 - E010
            # ------------------------------------------------

            c3 = rarest_address_candidates(
                country,
                address,
                rare_address_s2,
                rare_address_s3,
            )

            union.update(c3)

            recovered_after_rare_address += len(
                true_matches & union
            )

            # ------------------------------------------------
            # PASS 4 - E003
            # ------------------------------------------------

            c4 = exact_address_candidates(
                country,
                address,
                exact_address_s2,
                exact_address_s3,
            )

            union.update(c4)

            recovered_after_exact_address += len(
                true_matches & union
            )

            # ------------------------------------------------
            # FINAL UNION
            # ------------------------------------------------

            candidate_count = len(union)

            total_final_candidates += (
                candidate_count
            )

            if candidate_count > 0:
                entities_with_candidates += 1

            # Singleton statistics.
            if not true_matches:

                singleton_total += 1

                if candidate_count > 0:
                    singleton_with_candidates += 1

        print(
            f"Evaluated S1 chunk {chunk_no}"
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    recall_name_sorted = (
        recovered_after_name_sorted
        / total_true_links
    )

    recall_after_rare_name = (
        recovered_after_rare_name
        / total_true_links
    )

    recall_after_rare_address = (
        recovered_after_rare_address
        / total_true_links
    )

    recall_final = (
        recovered_after_exact_address
        / total_true_links
    )

    average_candidates = (
        total_final_candidates
        / total_s1
    )

    entity_candidate_rate = (
        entities_with_candidates
        / total_s1
    )

    singleton_candidate_rate = (
        singleton_with_candidates
        / singleton_total
    )

    estimated_total_pairs = (
        average_candidates * total_s1
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("E011 - MULTI-PASS BLOCKING RESULTS")
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
        "\nRecall progression:"
    )

    print(
        f"  After E004: "
        f"{recall_name_sorted:.4%}"
    )

    print(
        f"  After E007: "
        f"{recall_after_rare_name:.4%}"
    )

    print(
        f"  After E010: "
        f"{recall_after_rare_address:.4%}"
    )

    print(
        f"  After E003: "
        f"{recall_final:.4%}"
    )

    print(
        f"\nFinal recovered true links: "
        f"{recovered_after_exact_address:,}"
    )

    print(
        f"Final candidate recall: "
        f"{recall_final:.4%}"
    )

    print(
        f"Average candidates per S1: "
        f"{average_candidates:.4f}"
    )

    print(
        f"Estimated total candidate pairs: "
        f"{estimated_total_pairs:,.0f}"
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