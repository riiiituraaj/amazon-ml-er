from pathlib import Path
from collections import defaultdict

import pandas as pd

from normalization import extract_address_components


TRAIN_DIR = Path("dataset/train")

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

CHUNK_SIZE = 100_000

# Very common address-number keys create huge candidate buckets.
# We will not use them for blocking.
MAX_NUMBER_BUCKET = 500


def build_indexes(path: Path):
    """
    Build memory-conscious address indexes.

    We create:
        1. country + postal code
        2. country + address number

    Very common address-number buckets are excluded.
    """

    postal_index = defaultdict(set)
    number_index = defaultdict(set)

    print(f"\nBuilding indexes: {path.name}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_address",
                "country",
            ],
        ),
        start=1,
    ):

        for entity_id, address, country in zip(
            chunk["entity_id"],
            chunk["business_address"],
            chunk["country"],
        ):

            country_key = str(country).strip().lower()

            components = extract_address_components(address)

            # ---------------------------------------------------------------
            # Postal code
            # ---------------------------------------------------------------

            for postal in set(components["postal_codes"]):

                postal_index[
                    (country_key, postal)
                ].add(entity_id)

            # ---------------------------------------------------------------
            # Address number
            # ---------------------------------------------------------------

            for number in set(components["numbers"]):

                # Ignore extremely short numbers.
                if len(number) < 2:
                    continue

                number_index[
                    (country_key, number)
                ].add(entity_id)

        print(f"  processed chunk {chunk_number}")

    # Remove excessively common number buckets.
    print("\nFiltering common number buckets...")

    common_keys = [
        key
        for key, entity_ids in number_index.items()
        if len(entity_ids) > MAX_NUMBER_BUCKET
    ]

    for key in common_keys:
        del number_index[key]

    print(
        f"Postal keys retained: "
        f"{len(postal_index):,}"
    )

    print(
        f"Number keys retained: "
        f"{len(number_index):,}"
    )

    print(
        f"Number keys removed because bucket size > "
        f"{MAX_NUMBER_BUCKET}: "
        f"{len(common_keys):,}"
    )

    return postal_index, number_index


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
    s2_indexes,
    s3_indexes,
    ground_truth,
):

    s2_postal, s2_number = s2_indexes
    s3_postal, s3_number = s3_indexes

    total_entities = 0
    total_true_links = 0

    recovered_postal = 0
    recovered_number = 0
    recovered_union = 0

    total_postal_candidates = 0
    total_number_candidates = 0
    total_union_candidates = 0

    entities_with_postal = 0
    entities_with_number = 0
    entities_with_union = 0

    print("\nEvaluating address-component blocking...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            s1_file,
            sep="\t",
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_address",
                "country",
            ],
        ),
        start=1,
    ):

        for s1_id, address, country in zip(
            chunk["entity_id"],
            chunk["business_address"],
            chunk["country"],
        ):

            total_entities += 1

            truth = ground_truth.get(
                s1_id,
                set(),
            )

            total_true_links += len(truth)

            country_key = str(country).strip().lower()

            components = extract_address_components(address)

            # ---------------------------------------------------------------
            # Postal candidates
            # ---------------------------------------------------------------

            postal_candidates = set()

            for postal in set(
                components["postal_codes"]
            ):

                postal_candidates.update(
                    s2_postal.get(
                        (country_key, postal),
                        ()
                    )
                )

                postal_candidates.update(
                    s3_postal.get(
                        (country_key, postal),
                        ()
                    )
                )

            # ---------------------------------------------------------------
            # Number candidates
            # ---------------------------------------------------------------

            number_candidates = set()

            for number in set(
                components["numbers"]
            ):

                if len(number) < 2:
                    continue

                number_candidates.update(
                    s2_number.get(
                        (country_key, number),
                        ()
                    )
                )

                number_candidates.update(
                    s3_number.get(
                        (country_key, number),
                        ()
                    )
                )

            # ---------------------------------------------------------------
            # Union
            # ---------------------------------------------------------------

            union_candidates = (
                postal_candidates
                | number_candidates
            )

            # ---------------------------------------------------------------
            # Recall
            # ---------------------------------------------------------------

            recovered_postal += len(
                truth.intersection(
                    postal_candidates
                )
            )

            recovered_number += len(
                truth.intersection(
                    number_candidates
                )
            )

            recovered_union += len(
                truth.intersection(
                    union_candidates
                )
            )

            # ---------------------------------------------------------------
            # Counts
            # ---------------------------------------------------------------

            postal_count = len(postal_candidates)
            number_count = len(number_candidates)
            union_count = len(union_candidates)

            total_postal_candidates += postal_count
            total_number_candidates += number_count
            total_union_candidates += union_count

            if postal_count:
                entities_with_postal += 1

            if number_count:
                entities_with_number += 1

            if union_count:
                entities_with_union += 1

        print(
            f"Evaluated S1 chunk {chunk_number}"
        )

    def percentage(value, total):

        if total == 0:
            return 0.0

        return value / total * 100

    print("\n" + "=" * 80)
    print("E005 - ADDRESS COMPONENT BLOCKING RESULTS")
    print("=" * 80)

    print(
        f"\nS1 entities: "
        f"{total_entities:,}"
    )

    print(
        f"True links: "
        f"{total_true_links:,}"
    )

    print("\nCandidate Recall:")

    print(
        f"  Postal code: "
        f"{percentage(recovered_postal, total_true_links):.4f}%"
    )

    print(
        f"  Address number: "
        f"{percentage(recovered_number, total_true_links):.4f}%"
    )

    print(
        f"  UNION: "
        f"{percentage(recovered_union, total_true_links):.4f}%"
    )

    print("\nAverage candidates per S1:")

    print(
        f"  Postal code: "
        f"{total_postal_candidates / total_entities:.4f}"
    )

    print(
        f"  Address number: "
        f"{total_number_candidates / total_entities:.4f}"
    )

    print(
        f"  UNION: "
        f"{total_union_candidates / total_entities:.4f}"
    )

    print("\nS1 entities receiving candidates:")

    print(
        f"  Postal code: "
        f"{entities_with_postal:,} "
        f"({entities_with_postal / total_entities * 100:.2f}%)"
    )

    print(
        f"  Address number: "
        f"{entities_with_number:,} "
        f"({entities_with_number / total_entities * 100:.2f}%)"
    )

    print(
        f"  UNION: "
        f"{entities_with_union:,} "
        f"({entities_with_union / total_entities * 100:.2f}%)"
    )


def main():

    print("=" * 80)
    print("E005 - MEMORY-SAFE ADDRESS COMPONENT BLOCKING")
    print("=" * 80)

    print("\n1. Building S2 indexes")

    s2_indexes = build_indexes(
        S2_FILE
    )

    print("\n2. Building S3 indexes")

    s3_indexes = build_indexes(
        S3_FILE
    )

    print("\n3. Loading ground truth")

    ground_truth = load_ground_truth()

    print("\n4. Evaluating")

    evaluate_blocking(
        S1_FILE,
        s2_indexes,
        s3_indexes,
        ground_truth,
    )


if __name__ == "__main__":
    main()