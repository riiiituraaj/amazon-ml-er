from pathlib import Path
from collections import Counter

import pandas as pd


TRAIN_DIR = Path("dataset/train")
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

CHUNK_SIZE = 100_000


def analyze_ground_truth():
    total_s1 = 0

    singleton_count = 0
    matched_entity_count = 0

    total_positive_links = 0

    only_s2 = 0
    only_s3 = 0
    both_s2_s3 = 0

    match_distribution = Counter()

    s2_positive_links = 0
    s3_positive_links = 0

    duplicate_id_rows = 0

    print("Reading ground truth...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            GT_FILE,
            sep="\t",
            chunksize=CHUNK_SIZE,
            keep_default_na=False
        ),
        start=1
    ):

        for _, row in chunk.iterrows():

            total_s1 += 1

            raw_matches = row["matched_entity_ids"].strip()

            # Singleton
            if not raw_matches:
                singleton_count += 1
                match_distribution[0] += 1
                continue

            matches = [
                x.strip()
                for x in raw_matches.split(",")
                if x.strip()
            ]

            # Detect duplicate IDs within the same row
            if len(matches) != len(set(matches)):
                duplicate_id_rows += 1

            match_count = len(matches)

            matched_entity_count += 1
            total_positive_links += match_count

            match_distribution[match_count] += 1

            has_s2 = False
            has_s3 = False

            for entity_id in matches:

                if entity_id.startswith("S2-"):
                    s2_positive_links += 1
                    has_s2 = True

                elif entity_id.startswith("S3-"):
                    s3_positive_links += 1
                    has_s3 = True

            if has_s2 and has_s3:
                both_s2_s3 += 1

            elif has_s2:
                only_s2 += 1

            elif has_s3:
                only_s3 += 1

        print(f"Processed chunk {chunk_number}...")

    print("\n" + "=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    print(f"\nTotal S1 entities: {total_s1:,}")
    print(f"Singletons: {singleton_count:,}")
    print(f"Non-singletons: {matched_entity_count:,}")

    print(f"\nTotal positive links: {total_positive_links:,}")

    if total_s1:
        print(
            f"Average matches per S1: "
            f"{total_positive_links / total_s1:.4f}"
        )

    print("\nSource breakdown:")
    print(f"  S2 positive links: {s2_positive_links:,}")
    print(f"  S3 positive links: {s3_positive_links:,}")

    if total_positive_links:
        print(
            f"  S2 share: "
            f"{s2_positive_links / total_positive_links * 100:.2f}%"
        )
        print(
            f"  S3 share: "
            f"{s3_positive_links / total_positive_links * 100:.2f}%"
        )

    print("\nS1 match composition:")
    print(f"  Only S2: {only_s2:,}")
    print(f"  Only S3: {only_s3:,}")
    print(f"  Both S2 and S3: {both_s2_s3:,}")
    print(f"  Singleton: {singleton_count:,}")

    print("\nMatch-count distribution:")

    for count in sorted(match_distribution):
        entities = match_distribution[count]

        percentage = (
            entities / total_s1 * 100
            if total_s1
            else 0
        )

        print(
            f"  {count:>2} matches: "
            f"{entities:>10,} "
            f"({percentage:6.2f}%)"
        )

    print(
        "\nRows containing duplicate matched IDs: "
        f"{duplicate_id_rows:,}"
    )


if __name__ == "__main__":
    analyze_ground_truth()