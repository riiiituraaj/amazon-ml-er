from pathlib import Path
from collections import defaultdict

import pandas as pd

from normalization import normalize_name


TRAIN_DIR = Path("dataset/train")

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

CHUNK_SIZE = 100_000


def build_name_index(path: Path):
    """
    Build:
        normalized_name -> set(entity_id)

    The file is processed in chunks to keep memory usage reasonable.
    """

    index = defaultdict(set)

    print(f"\nBuilding name index: {path.name}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            usecols=["entity_id", "business_name"],
        ),
        start=1,
    ):

        for entity_id, business_name in zip(
            chunk["entity_id"],
            chunk["business_name"],
        ):

            normalized = normalize_name(business_name)

            if normalized:
                index[normalized].add(entity_id)

        print(f"  processed chunk {chunk_number}")

    print(f"Index size: {len(index):,} unique normalized names")

    return index


def load_ground_truth():
    """
    Load ground truth into:

        S1 ID -> set of true matched IDs
    """

    truth = {}

    print("\nLoading ground truth...")

    for chunk in pd.read_csv(
        GT_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
        usecols=["source1_entity_id", "matched_entity_ids"],
    ):

        for s1_id, raw_matches in zip(
            chunk["source1_entity_id"],
            chunk["matched_entity_ids"],
        ):

            if raw_matches.strip():
                matches = {
                    x.strip()
                    for x in raw_matches.split(",")
                    if x.strip()
                }
            else:
                matches = set()

            truth[s1_id] = matches

    return truth


def evaluate_exact_name_blocking(
    s1_file: Path,
    s2_index,
    s3_index,
    ground_truth,
):
    """
    Evaluate candidate recall of exact normalized-name blocking.
    """

    total_true_links = 0
    recovered_true_links = 0

    total_entities = 0
    entities_with_candidates = 0

    total_candidates = 0

    singleton_entities = 0
    singleton_with_candidates = 0

    print("\nEvaluating exact normalized-name blocking...")

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

            truth = ground_truth.get(s1_id, set())

            if not truth:
                singleton_entities += 1

            normalized_name = normalize_name(business_name)

            candidates = set()

            if normalized_name:
                candidates.update(
                    s2_index.get(normalized_name, set())
                )

                candidates.update(
                    s3_index.get(normalized_name, set())
                )

            candidate_count = len(candidates)

            total_candidates += candidate_count

            if candidate_count > 0:
                entities_with_candidates += 1

                if not truth:
                    singleton_with_candidates += 1

            total_true_links += len(truth)

            recovered_true_links += len(
                truth.intersection(candidates)
            )

        print(f"Evaluated S1 chunk {chunk_number}")

    candidate_recall = (
        recovered_true_links / total_true_links
        if total_true_links
        else 0
    )

    avg_candidates = (
        total_candidates / total_entities
        if total_entities
        else 0
    )

    print("\n" + "=" * 80)
    print("EXACT NORMALIZED NAME BLOCKING RESULTS")
    print("=" * 80)

    print(f"\nS1 entities: {total_entities:,}")
    print(f"True links: {total_true_links:,}")
    print(f"Recovered true links: {recovered_true_links:,}")

    print(
        f"\nCandidate Recall: "
        f"{candidate_recall * 100:.4f}%"
    )

    print(
        f"Average candidates per S1: "
        f"{avg_candidates:.4f}"
    )

    print(
        f"Entities with >=1 candidate: "
        f"{entities_with_candidates:,} "
        f"({entities_with_candidates / total_entities * 100:.2f}%)"
    )

    print(
        f"\nSingletons: "
        f"{singleton_entities:,}"
    )

    print(
        f"Singletons receiving candidates: "
        f"{singleton_with_candidates:,} "
        f"({singleton_with_candidates / singleton_entities * 100:.2f}%)"
        if singleton_entities
        else "Singletons receiving candidates: 0"
    )


def main():

    print("=" * 80)
    print("EXACT NAME BLOCKING EXPERIMENT")
    print("=" * 80)

    # Build indexes separately because S2 and S3 are different sources.
    print("\n1. Building S2 index")
    s2_index = build_name_index(S2_FILE)

    print("\n2. Building S3 index")
    s3_index = build_name_index(S3_FILE)

    print("\n3. Loading ground truth")
    ground_truth = load_ground_truth()

    print("\n4. Evaluating blocker")

    evaluate_exact_name_blocking(
        S1_FILE,
        s2_index,
        s3_index,
        ground_truth,
    )


if __name__ == "__main__":
    main()