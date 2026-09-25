import os
from collections import defaultdict

import pandas as pd

from normalization import normalize_name


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

NGRAM_SIZE = 3

MIN_TOKEN_LEN = 3

# Ignore extremely common character n-grams.
MAX_BUCKET_SIZE = 100

# Only use the rarest few n-grams from each S1 name.
TOP_K_NGRAMS = 4


# ============================================================
# CHARACTER N-GRAMS
# ============================================================

def char_ngrams(text, n=NGRAM_SIZE):
    """
    Generate character n-grams from a normalized string.

    Spaces are retained as separators so that we capture
    useful boundaries such as:

        "family office"
    """

    if not text:
        return set()

    padded = f"  {text}  "

    return {
        padded[i:i + n]
        for i in range(len(padded) - n + 1)
    }


# ============================================================
# BUILD INDEX
# ============================================================

def build_char_ngram_index(path):

    print(
        f"Building character n-gram index: "
        f"{os.path.basename(path)}"
    )

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
            name = row.business_name

            if not name:
                continue

            normalized = normalize_name(name)

            if not normalized:
                continue

            # Ignore extremely short names.
            if len(normalized.replace(" ", "")) < MIN_TOKEN_LEN:
                continue

            grams = char_ngrams(normalized)

            for gram in grams:
                buckets[(country, gram)].append(entity_id)

    print("\nFiltering common character n-grams...")

    index = {}
    removed = 0

    for key, entity_ids in buckets.items():

        if len(entity_ids) <= MAX_BUCKET_SIZE:

            index[key] = entity_ids

        else:

            removed += 1

    print(
        f"Retained n-gram buckets: "
        f"{len(index):,}"
    )

    print(
        f"Removed common n-gram buckets: "
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
# CANDIDATES
# ============================================================

def get_candidates(
    country,
    name,
    index_s2,
    index_s3,
):

    if not name:
        return set()

    normalized = normalize_name(name)

    if not normalized:
        return set()

    grams = char_ngrams(normalized)

    if not grams:
        return set()

    # Determine rarity from the size of the available buckets.
    gram_info = []

    for gram in grams:

        key = (country, gram)

        count_s2 = len(index_s2.get(key, []))
        count_s3 = len(index_s3.get(key, []))

        counts = []

        if count_s2:
            counts.append(count_s2)

        if count_s3:
            counts.append(count_s3)

        if not counts:
            continue

        rarity = min(counts)

        gram_info.append(
            (rarity, gram)
        )

    gram_info.sort(
        key=lambda x: x[0]
    )

    selected = [
        gram
        for _, gram in gram_info[:TOP_K_NGRAMS]
    ]

    candidates = set()

    for gram in selected:

        key = (country, gram)

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
    print("E012 - RARE CHARACTER N-GRAM BLOCKING")
    print("=" * 80)

    print(
        f"\nN-gram size: {NGRAM_SIZE}"
    )

    print(
        f"Maximum bucket size: "
        f"{MAX_BUCKET_SIZE}"
    )

    print(
        f"Top n-grams per S1: "
        f"{TOP_K_NGRAMS}"
    )

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    print("\n1. Building S2 character n-gram index")

    index_s2 = build_char_ngram_index(
        S2_FILE
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    print("\n2. Building S3 character n-gram index")

    index_s3 = build_char_ngram_index(
        S3_FILE
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print("\n3. Loading ground truth")

    ground_truth = load_ground_truth(
        GT_FILE
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n4. Evaluating blocker")
    print(
        "Evaluating rare character n-gram blocking..."
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
            name = row.business_name

            true_matches = ground_truth.get(
                s1_id,
                set()
            )

            candidates = get_candidates(
                country,
                name,
                index_s2,
                index_s3,
            )

            candidate_count = len(candidates)

            total_s1 += 1

            total_true_links += len(
                true_matches
            )

            if candidate_count > 0:

                entities_with_candidates += 1
                total_candidates += candidate_count

            recovered_links += len(
                true_matches & candidates
            )

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
        singleton_with_candidates
        / singleton_total
        if singleton_total
        else 0.0
    )

    estimated_total_pairs = (
        average_candidates * total_s1
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("E012 - RARE CHARACTER N-GRAM BLOCKING RESULTS")
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
        f"{candidate_recall:.4%}"
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