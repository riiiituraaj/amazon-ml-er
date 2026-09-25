from pathlib import Path

import pandas as pd


TRAIN_DIR = Path("dataset/train")

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

CHUNK_SIZE = 100_000
SAMPLE_SIZE = 30


def get_sample_ground_truth():

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        keep_default_na=False,
    )

    matched = gt[
        gt["matched_entity_ids"].str.strip() != ""
    ]

    return matched.sample(
        n=min(SAMPLE_SIZE, len(matched)),
        random_state=42,
    )


def collect_ids(sample):

    s1_ids = set(sample["source1_entity_id"])

    s2_ids = set()
    s3_ids = set()

    for raw in sample["matched_entity_ids"]:

        for entity_id in raw.split(","):

            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                s3_ids.add(entity_id)

    return s1_ids, s2_ids, s3_ids


def find_records(path, wanted_ids):

    results = {}

    if not wanted_ids:
        return results

    print(f"\nSearching {path.name}...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ),
        start=1,
    ):

        matches = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        for _, row in matches.iterrows():
            results[row["entity_id"]] = row.to_dict()

        print(f"  processed chunk {chunk_number}")

        if len(results) == len(wanted_ids):
            break

    return results


def main():

    print("=" * 80)
    print("E002 - REAL MATCH INSPECTION")
    print("=" * 80)

    sample = get_sample_ground_truth()

    s1_ids, s2_ids, s3_ids = collect_ids(sample)

    print(f"\nSampled S1 entities: {len(s1_ids)}")
    print(f"S2 matches to retrieve: {len(s2_ids)}")
    print(f"S3 matches to retrieve: {len(s3_ids)}")

    s1_records = find_records(
        S1_FILE,
        s1_ids,
    )

    s2_records = find_records(
        S2_FILE,
        s2_ids,
    )

    s3_records = find_records(
        S3_FILE,
        s3_ids,
    )

    print("\n" + "=" * 80)
    print("REAL MATCH EXAMPLES")
    print("=" * 80)

    for _, row in sample.iterrows():

        s1_id = row["source1_entity_id"]

        print("\n" + "-" * 80)

        s1 = s1_records.get(s1_id)

        if not s1:
            continue

        print(f"S1: {s1_id}")
        print(f"Name:    {s1['business_name']}")
        print(f"Address: {s1['business_address']}")
        print(f"Country: {s1['country']}")

        print("\nTRUE MATCHES:")

        for entity_id in row["matched_entity_ids"].split(","):

            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                record = s2_records.get(entity_id)

            else:
                record = s3_records.get(entity_id)

            if not record:
                continue

            print(f"\n  {entity_id}")
            print(f"  Name:    {record['business_name']}")
            print(f"  Address: {record['business_address']}")
            print(f"  Country: {record['country']}")


if __name__ == "__main__":
    main()