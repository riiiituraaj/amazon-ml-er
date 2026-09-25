from pathlib import Path
from collections import defaultdict

import pandas as pd

from normalization import normalize_name_v2


TRAIN_DIR = Path("dataset/train")

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

CHUNK_SIZE = 100_000

# A token appearing in more than this many records
# is considered too common for blocking.
MAX_BUCKET_SIZE = 300

# Ignore extremely short tokens.
MIN_TOKEN_LENGTH = 3


def get_tokens(name: str):
    """
    Return unique normalized name tokens.
    """

    normalized = normalize_name_v2(name)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= MIN_TOKEN_LENGTH
    }


def build_token_index(path: Path):
    """
    Build:

        (country, token) -> entity IDs

    Common tokens with huge buckets are removed.
    """

    index = defaultdict(set)

    print(f"\nBuilding token index: {path.name}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_name",
                "country",
            ],
        ),
        start=1,
    ):

        for entity_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"],
        ):

            country_key = (
                str(country).strip().lower()
            )

            tokens = get_tokens(name)

            for token in tokens:
                index[
                    (country_key, token)
                ].add(entity_id)

        print(
            f"  processed chunk {chunk_number}"
        )

    print("\nFiltering common tokens...")

    oversized = []

    for key, entity_ids in index.items():
        if len(entity_ids) > MAX_BUCKET_SIZE:
            oversized.append(key)

    for key in oversized:
        del index[key]

    print(
        f"Retained token buckets: "
        f"{len(index):,}"
    )

    print(
        f"Removed common token buckets: "
        f"{len(oversized):,}"
    )

    return index


def load_ground_truth():

    truth = {}

    print("\nLoading ground truth...")

    for chunk in pd.read_csv(
        GT_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    ):

        for s1_id, raw_matches in zip(
            chunk["source1_entity_id"],
            chunk["matched_entity_ids"],
        ):

            if raw_matches.strip():

                matches = {
                    entity_id.strip()
                    for entity_id in raw_matches.split(",")
                    if entity_id.strip()
                }

            else:
                matches = set()

            truth[s1_id] = matches

    return truth


def evaluate_blocking(
    s1_file,
    s2_index,
    s3_index,
    ground_truth,
):

    total_entities = 0
    total_true_links = 0
    recovered_true_links = 0

    total_candidates = 0
    entities_with_candidates = 0

    singleton_entities = 0
    singleton_with_candidates = 0

    print("\nEvaluating rare-token blocking...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            s1_file,
            sep="\t",
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_name",
                "country",
            ],
        ),
        start=1,
    ):

        for s1_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"],
        ):

            total_entities += 1

            truth = ground_truth.get(
                s1_id,
                set(),
            )

            if not truth:
                singleton_entities += 1

            total_true_links += len(truth)

            country_key = (
                str(country).strip().lower()
            )

            tokens = get_tokens(name)

            candidates = set()

            for token in tokens:

                key = (
                    country_key,
                    token,
                )

                candidates.update(
                    s2_index.get(key, ())
                )

                candidates.update(
                    s3_index.get(key, ())
                )

            candidate_count = len(candidates)

            total_candidates += candidate_count

            if candidate_count > 0:

                entities_with_candidates += 1

                if not truth:
                    singleton_with_candidates += 1

            recovered_true_links += len(
                truth.intersection(candidates)
            )

        print(
            f"Evaluated S1 chunk {chunk_number}"
        )

    candidate_recall = (
        recovered_true_links / total_true_links
        if total_true_links
        else 0.0
    )

    average_candidates = (
        total_candidates / total_entities
        if total_entities
        else 0.0
    )

    entity_coverage = (
        entities_with_candidates / total_entities * 100
        if total_entities
        else 0.0
    )

    singleton_rate = (
        singleton_with_candidates /
        singleton_entities * 100
        if singleton_entities
        else 0.0
    )

    print("\n" + "=" * 80)
    print("E007 - RARE NAME TOKEN BLOCKING RESULTS")
    print("=" * 80)

    print(
        f"\nS1 entities: "
        f"{total_entities:,}"
    )

    print(
        f"True links: "
        f"{total_true_links:,}"
    )

    print(
        f"Recovered true links: "
        f"{recovered_true_links:,}"
    )

    print(
        f"\nCandidate Recall: "
        f"{candidate_recall * 100:.4f}%"
    )

    print(
        f"Average candidates per S1: "
        f"{average_candidates:.4f}"
    )

    print(
        f"Entities with >=1 candidate: "
        f"{entities_with_candidates:,} "
        f"({entity_coverage:.2f}%)"
    )

    print(
        f"\nSingletons: "
        f"{singleton_entities:,}"
    )

    print(
        f"Singletons receiving candidates: "
        f"{singleton_with_candidates:,} "
        f"({singleton_rate:.2f}%)"
    )


def main():

    print("=" * 80)
    print("E007 - RARE NAME TOKEN BLOCKING")
    print("=" * 80)

    print("\n1. Building S2 token index")

    s2_index = build_token_index(
        S2_FILE
    )

    print("\n2. Building S3 token index")

    s3_index = build_token_index(
        S3_FILE
    )

    print("\n3. Loading ground truth")

    ground_truth = load_ground_truth()

    print("\n4. Evaluating blocker")

    evaluate_blocking(
        S1_FILE,
        s2_index,
        s3_index,
        ground_truth,
    )


if __name__ == "__main__":
    main()