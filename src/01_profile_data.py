from pathlib import Path
from collections import Counter
import pandas as pd

TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")

CHUNK_SIZE = 100_000


def profile_file(path: Path):
    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    # Read a tiny sample to inspect structure
    sample = pd.read_csv(path, sep="\t", nrows=5)

    print("\nColumns:")
    for col in sample.columns:
        print(f"  - {col}")

    print("\nSample:")
    print(sample.to_string(index=False))

    print("\nCounting rows and missing values...")

    total_rows = 0
    missing = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):
        total_rows += len(chunk)

        for col in chunk.columns:
            missing[col] += chunk[col].isna().sum()

    print(f"\nTotal rows: {total_rows:,}")

    print("\nMissing values:")
    for col, count in missing.items():
        percentage = (count / total_rows) * 100
        print(f"  {col}: {count:,} ({percentage:.2f}%)")

    return total_rows


def profile_ground_truth(path: Path):
    print("\n" + "=" * 80)
    print(f"GROUND TRUTH: {path}")
    print("=" * 80)

    total_entities = 0
    singleton_count = 0
    match_count_distribution = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):
        total_entities += len(chunk)

        for matches in chunk["matched_entity_ids"]:
            if not matches:
                count = 0
            else:
                count = len(
                    [x for x in matches.split(",") if x.strip()]
                )

            match_count_distribution[count] += 1

            if count == 0:
                singleton_count += 1

    print(f"\nTotal Source 1 entities: {total_entities:,}")
    print(f"Singletons: {singleton_count:,}")

    if total_entities:
        print(
            f"Singleton percentage: "
            f"{singleton_count / total_entities * 100:.2f}%"
        )

    print("\nMatches per Source 1 entity:")

    for count in sorted(match_count_distribution):
        entities = match_count_distribution[count]
        percentage = entities / total_entities * 100

        print(
            f"  {count:>3} matches: "
            f"{entities:>10,} entities "
            f"({percentage:6.2f}%)"
        )


def profile_countries(path: Path):
    print("\n" + "=" * 80)
    print(f"COUNTRIES: {path}")
    print("=" * 80)

    countries = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):
        countries.update(
            chunk["country"].astype(str).str.strip()
        )

    total = sum(countries.values())

    for country, count in countries.most_common():
        print(
            f"  {country!r}: "
            f"{count:,} "
            f"({count / total * 100:.2f}%)"
        )


def main():

    train_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ]

    test_files = [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]

    print("\n\nTRAINING DATA")
    print("=" * 80)

    for filename in train_files:
        profile_file(TRAIN_DIR / filename)

    profile_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    print("\n\nTRAINING COUNTRIES")
    print("=" * 80)

    for filename in train_files:
        profile_countries(TRAIN_DIR / filename)

    print("\n\nTEST DATA")
    print("=" * 80)

    for filename in test_files:
        profile_file(TEST_DIR / filename)

    print("\n\nTEST COUNTRIES")
    print("=" * 80)

    for filename in test_files:
        profile_countries(TEST_DIR / filename)


if __name__ == "__main__":
    main()
    