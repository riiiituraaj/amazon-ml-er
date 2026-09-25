from pathlib import Path
from collections import defaultdict

import pandas as pd

from normalization import normalize_name_token_sorted


TRAIN_DIR = Path("dataset/train")

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

CHUNK_SIZE = 100_000

# Maximum number of records allowed in one blocking bucket.
# Extremely common prefixes are discarded because they are
# not useful for candidate generation.
MAX_BUCKET_SIZE = 500

# Number of characters used from the compact token-sorted name.
PREFIX_LENGTH = 6


def make_name_prefix(name: str) -> str:
    """
    Create a character-prefix blocking key.

    Pipeline:

        business name
            ↓
        normalized name
            ↓
        token sorting
            ↓
        remove spaces
            ↓
        first 6 characters

    Example:

        "Desert Society Inc"
            ->
        "desert inc society"
            ->
        "desertincsociety"
            ->
        "desert"
    """

    normalized = normalize_name_token_sorted(name)

    if not normalized:
        return ""

    compact = normalized.replace(" ", "")

    if len(compact) < PREFIX_LENGTH:
        return compact

    return compact[:PREFIX_LENGTH]


def add_to_index(
    index,
    oversized_keys,
    key,
    entity_id,
):
    """
    Add an entity to a blocking bucket.

    If a bucket becomes too large, permanently discard
    that key because it is not discriminative enough.
    """

    if not key:
        return

    if key in oversized_keys:
        return

    if key not in index:
        index[key] = {entity_id}
        return

    bucket = index[key]

    bucket.add(entity_id)

    if len(bucket) > MAX_BUCKET_SIZE:
        del index[key]
        oversized_keys.add(key)


def build_prefix_index(path: Path):
    """
    Build:

        (country, name_prefix) -> entity IDs

    Common prefixes with more than MAX_BUCKET_SIZE records
    are discarded.
    """

    index = {}
    oversized_keys = set()

    print(
        f"\nBuilding name-prefix index: {path.name}"
    )

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

        for entity_id, business_name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"],
        ):

            prefix = make_name_prefix(
                business_name
            )

            if not prefix:
                continue

            country_key = (
                str(country)
                .strip()
                .lower()
            )

            key = (
                country_key,
                prefix,
            )

            add_to_index(
                index,
                oversized_keys,
                key,
                entity_id,
            )

        print(
            f"  processed chunk {chunk_number}"
        )

    print(
        f"Retained prefix buckets: "
        f"{len(index):,}"
    )

    print(
        f"Discarded oversized prefixes: "
        f"{len(oversized_keys):,}"
    )

    return index


def load_ground_truth():
    """
    Load:

        S1 ID -> set of true matched IDs
    """

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
    """
    Evaluate name-prefix blocking.
    """

    total_entities = 0
    total_true_links = 0
    recovered_true_links = 0

    total_candidates = 0
    entities_with_candidates = 0

    singleton_entities = 0
    singleton_with_candidates = 0

    print(
        "\nEvaluating name-prefix blocking..."
    )

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

        for s1_id, business_name, country in zip(
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

            prefix = make_name_prefix(
                business_name
            )

            candidates = set()

            if prefix:

                country_key = (
                    str(country)
                    .strip()
                    .lower()
                )

                key = (
                    country_key,
                    prefix,
                )

                candidates.update(
                    s2_index.get(
                        key,
                        ()
                    )
                )

                candidates.update(
                    s3_index.get(
                        key,
                        ()
                    )
                )

            candidate_count = len(
                candidates
            )

            total_candidates += candidate_count

            if candidate_count > 0:

                entities_with_candidates += 1

                if not truth:
                    singleton_with_candidates += 1

            recovered_true_links += len(
                truth.intersection(
                    candidates
                )
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
        entities_with_candidates / total_entities
        if total_entities
        else 0.0
    )

    singleton_candidate_rate = (
        singleton_with_candidates /
        singleton_entities
        if singleton_entities
        else 0.0
    )

    print("\n" + "=" * 80)
    print(
        "E006 - CHARACTER PREFIX NAME BLOCKING RESULTS"
    )
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
        f"({entity_coverage * 100:.2f}%)"
    )

    print(
        f"\nSingletons: "
        f"{singleton_entities:,}"
    )

    print(
        f"Singletons receiving candidates: "
        f"{singleton_with_candidates:,} "
        f"({singleton_candidate_rate * 100:.2f}%)"
    )


def main():

    print("=" * 80)
    print(
        "E006 - CHARACTER PREFIX NAME BLOCKING"
    )
    print("=" * 80)

    print("\n1. Building S2 prefix index")

    s2_index = build_prefix_index(
        S2_FILE
    )

    print("\n2. Building S3 prefix index")

    s3_index = build_prefix_index(
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