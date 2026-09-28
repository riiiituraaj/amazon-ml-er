from __future__ import annotations

"""
PRODUCTION TEST INFERENCE
=========================

Amazon ML Challenge - Business Entity Resolution

Pipeline:

    TEST SOURCE 1
         |
         v
    Blocking indexes
         |
         v
    Final Top-K candidate set
         |
         +------------------------------+
         |                              |
         v                              v
candidate_pairs.tsv               Pair features
                                       |
                                       v
                                  LightGBM model
                                       |
                                       v
                              Isotonic calibration
                                       |
                                       v
                               threshold decoding
                                       |
                                       v
                              matching_results.tsv

Important:
- Uses the same blocking functions as training.
- Uses the same make_pair_features() as training.
- blocking_scores is passed as:
      {candidate_id: blocking_score}
- candidate_pairs.tsv is written BEFORE ML prediction.
- The trained threshold from competitive_lgbm.joblib is used.
- The model is NOT retrained here.
- Processing is chunked to control memory usage.
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================

import argparse
import csv
import gc
import heapq
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# ============================================================================
# THIRD-PARTY
# ============================================================================

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# PATH SETUP
# ============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from train_competitive_model import (
    FEATURE_NAMES,
    load_source_store,
    make_pair_features,
)

from signal_ranked_blocking import (
    FALLBACK_THRESHOLD,
    add_broad_fallback_scores,
    build_address_token_index,
    build_exact_address_index,
    build_name_token_index,
    build_token_sorted_name_index,
    score_candidates,
)


# ============================================================================
# FILE PATHS
# ============================================================================

TEST_DIR = PROJECT_DIR / "dataset" / "test"

S1_FILE = TEST_DIR / "test_source1.tsv"
S2_FILE = TEST_DIR / "test_source2.tsv"
S3_FILE = TEST_DIR / "test_source3.tsv"

MODEL_FILE = (
    PROJECT_DIR
    / "models"
    / "competitive_lgbm.joblib"
)

OUTPUT_DIR = PROJECT_DIR / "output"

MATCHING_FILE = (
    OUTPUT_DIR
    / "matching_results.tsv"
)

CANDIDATE_FILE = (
    OUTPUT_DIR
    / "candidate_pairs.tsv"
)

# Temporary files are used so a failed inference does not destroy a previous
# successful submission.
TEMP_MATCHING_FILE = (
    OUTPUT_DIR
    / "matching_results.tmp.tsv"
)

TEMP_CANDIDATE_FILE = (
    OUTPUT_DIR
    / "candidate_pairs.tmp.tsv"
)


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Small chunk keeps RAM under control.
#
# Each S1 entity can have up to TOP-K candidates. With K=40 and 36 features,
# 250 S1 rows is only about 10,000 candidate feature rows at maximum.
S1_CHUNK_SIZE = 250

DEFAULT_THRESHOLD = 0.55
DEFAULT_TOP_K = 40

PROGRESS_EVERY = 10_000


# ============================================================================
# CLI
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained LightGBM entity-resolution model "
            "on the complete test dataset."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Build indexes/stores and run a small end-to-end "
            "feature/model test without creating final output files."
        ),
    )

    return parser.parse_args()


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def safe_text(value) -> str:
    """
    Convert a value into a clean string.

    Prevents strings such as 'nan' from entering the feature pipeline.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def safe_remove(path: Path) -> None:
    """
    Safely remove a file if it exists.
    """

    if not path.exists():
        return

    try:
        path.unlink()
    except PermissionError as exc:
        raise PermissionError(
            f"Could not remove {path}. "
            "Close the file if it is open in Excel, VS Code, etc."
        ) from exc


def check_required_files() -> None:
    """
    Make sure all required inputs and the trained model exist.
    """

    required = [
        S1_FILE,
        S2_FILE,
        S3_FILE,
        MODEL_FILE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        print("\nERROR: Missing required files:")

        for path in missing:
            print(f"  {path}")

        raise FileNotFoundError(
            "One or more required inference files are missing."
        )


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_artifact():
    """
    Load:

        LightGBM model
        isotonic calibrator
        tuned threshold
        candidate K
        feature metadata
    """

    print("\n" + "=" * 80)
    print("LOADING TRAINED MODEL")
    print("=" * 80)

    print(
        f"\nModel file:\n  {MODEL_FILE}"
    )

    artifact = joblib.load(
        MODEL_FILE
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "competitive_lgbm.joblib does not contain "
            "the expected dictionary artifact."
        )

    model = artifact.get(
        "model"
    )

    calibrator = artifact.get(
        "calibrator"
    )

    threshold = float(
        artifact.get(
            "threshold",
            DEFAULT_THRESHOLD,
        )
    )

    config = (
        artifact.get(
            "config"
        )
        or {}
    )

    # Training code stored candidate-top-k information under one of these
    # fields. Keep multiple fallbacks for compatibility.
    top_k = int(
        config.get(
            "candidate_top_k",
            config.get(
                "train_top_k",
                DEFAULT_TOP_K,
            ),
        )
    )

    saved_features = artifact.get(
        "feature_names",
        FEATURE_NAMES,
    )

    if model is None:
        raise ValueError(
            "Saved model artifact does not contain 'model'."
        )

    if calibrator is None:
        raise ValueError(
            "Saved model artifact does not contain 'calibrator'."
        )

    # ------------------------------------------------------------------------
    # IMPORTANT FEATURE CONSISTENCY CHECK
    # ------------------------------------------------------------------------

    if list(saved_features) != list(
        FEATURE_NAMES
    ):
        raise ValueError(
            "\nFEATURE MISMATCH\n"
            "The inference code is not using the same feature order "
            "as the model.\n\n"
            f"Saved model features : {len(saved_features)}\n"
            f"Current features     : {len(FEATURE_NAMES)}\n\n"
            "Do not proceed until training/inference feature definitions "
            "are synchronized."
        )

    # LightGBM sklearn wrapper normally exposes this after fitting.
    model_feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (
        model_feature_count is not None
        and int(model_feature_count)
        != len(FEATURE_NAMES)
    ):
        raise ValueError(
            "\nLIGHTGBM FEATURE COUNT MISMATCH\n"
            f"Model expects : {model_feature_count}\n"
            f"Code provides : {len(FEATURE_NAMES)}"
        )

    print(
        f"\nThreshold       : {threshold:.6f}"
    )

    print(
        f"Candidate Top-K : {top_k}"
    )

    print(
        f"Feature count   : {len(FEATURE_NAMES)}"
    )

    print(
        "Model artifact  : OK"
    )

    return (
        model,
        calibrator,
        threshold,
        top_k,
    )


# ============================================================================
# TEST SOURCE-1 READER
# ============================================================================

def source1_reader():
    """
    Stream Source-1 in small chunks.

    dtype=str:
        Keeps entity IDs as strings.

    keep_default_na=False:
        Missing text fields become empty strings rather than NaN.
    """

    return pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        chunksize=S1_CHUNK_SIZE,
    )


# ============================================================================
# BLOCKING
# ============================================================================

def generate_candidates(
    country: str,
    name: str,
    address: str,
    indexes: tuple,
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    Generate the FINAL pre-ML candidate set.

    Return value:

        [
            (candidate_id, blocking_score),
            ...
        ]

    Candidate ranking happens before feature generation/model scoring.
    """

    (
        name_exact_s2,
        name_exact_s3,
        name_token_s2,
        name_token_s3,
        address_token_s2,
        address_token_s3,
        exact_address_s2,
        exact_address_s3,
    ) = indexes

    # ------------------------------------------------------------------------
    # Primary blocking
    # ------------------------------------------------------------------------

    scores = score_candidates(
        country,
        name,
        address,
        name_exact_s2,
        name_exact_s3,
        name_token_s2,
        name_token_s3,
        address_token_s2,
        address_token_s3,
        exact_address_s2,
        exact_address_s3,
    )

    # ------------------------------------------------------------------------
    # Adaptive fallback
    # ------------------------------------------------------------------------

    if len(scores) <= FALLBACK_THRESHOLD:
        add_broad_fallback_scores(
            scores,
            country,
            address,
            address_token_s2,
            address_token_s3,
        )

    if not scores:
        return []

    # ------------------------------------------------------------------------
    # FINAL PRE-ML TOP-K PRUNING
    # ------------------------------------------------------------------------

    ranked = heapq.nlargest(
        top_k,
        scores.items(),
        key=lambda item: (
            float(item[1]),
            str(item[0]),
        ),
    )

    # Defensive duplicate removal.
    final_candidates = []

    seen = set()

    for candidate_id, score in ranked:

        candidate_id = str(
            candidate_id
        )

        if candidate_id in seen:
            continue

        seen.add(
            candidate_id
        )

        final_candidates.append(
            (
                candidate_id,
                float(score),
            )
        )

    return final_candidates


# ============================================================================
# FEATURE GENERATION
# ============================================================================

def build_feature_matrix(
    country: str,
    name: str,
    address: str,
    ranked_candidates: Sequence[Tuple[str, float]],
    target_store: Dict,
) -> np.ndarray:
    """
    Generate features for one Source-1 entity.

    CRITICAL INTERFACE:

        make_pair_features(
            s1_country,
            s1_name_raw,
            s1_address_raw,
            candidate_ids,
            blocking_scores,
            target_store,
        )

    blocking_scores MUST be:

        {
            candidate_id: score
        }

    and NOT:

        [score1, score2, score3, ...]
    """

    number_of_candidates = len(
        ranked_candidates
    )

    feature_count = len(
        FEATURE_NAMES
    )

    if number_of_candidates == 0:
        return np.empty(
            (
                0,
                feature_count,
            ),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------------
    # Candidate IDs
    # ------------------------------------------------------------------------

    candidate_ids = [
        str(candidate_id)
        for candidate_id, _score
        in ranked_candidates
    ]

    # ------------------------------------------------------------------------
    # CRITICAL FIX:
    # dictionary mapping candidate ID -> blocking score
    # ------------------------------------------------------------------------

    blocking_scores = {
        str(candidate_id): float(score)
        for candidate_id, score
        in ranked_candidates
    }

    # ------------------------------------------------------------------------
    # Feature-store integrity
    # ------------------------------------------------------------------------

    missing = [
        candidate_id
        for candidate_id
        in candidate_ids
        if candidate_id
        not in target_store
    ]

    if missing:
        preview = ", ".join(
            missing[:10]
        )

        raise KeyError(
            "\nCandidate IDs generated by the blocker "
            "were not found in target_store.\n"
            f"Missing IDs: {len(missing)}\n"
            f"Examples: {preview}"
        )

    # ------------------------------------------------------------------------
    # USE THE EXACT TRAINING FEATURE FUNCTION
    # ------------------------------------------------------------------------

    features = make_pair_features(
        country,
        name,
        address,
        candidate_ids,
        blocking_scores,
        target_store,
    )

    X = np.asarray(
        features,
        dtype=np.float32,
    )

    # A single candidate can occasionally come back as a 1-D array.
    if X.ndim == 1:
        X = X.reshape(
            1,
            -1,
        )

    expected_shape = (
        number_of_candidates,
        feature_count,
    )

    if X.shape != expected_shape:
        raise ValueError(
            "\nFEATURE MATRIX SHAPE ERROR\n"
            f"Expected : {expected_shape}\n"
            f"Received : {X.shape}"
        )

    if not np.isfinite(
        X
    ).all():

        bad_values = int(
            (~np.isfinite(X)).sum()
        )

        raise ValueError(
            "\nFeature matrix contains "
            f"{bad_values} NaN/Inf values."
        )

    return X


# ============================================================================
# MODEL PREDICTION
# ============================================================================

def predict_features(
    X: np.ndarray,
    model,
    calibrator,
) -> np.ndarray:
    """
    Run:

        LightGBM probability
             ->
        Isotonic calibration

    Returns calibrated probability for each candidate row.
    """

    if X.size == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2-D feature matrix, got {X.ndim}-D."
        )

    if X.shape[1] != len(
        FEATURE_NAMES
    ):
        raise ValueError(
            "\nPrediction matrix has wrong number of features.\n"
            f"Expected: {len(FEATURE_NAMES)}\n"
            f"Received: {X.shape[1]}"
        )

    # ------------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------------

    raw_probabilities = (
        model.predict_proba(X)[:, 1]
    )

    raw_probabilities = np.asarray(
        raw_probabilities,
        dtype=np.float32,
    )

    if not np.isfinite(
        raw_probabilities
    ).all():
        raise ValueError(
            "LightGBM produced NaN/Inf probabilities."
        )

    # ------------------------------------------------------------------------
    # Isotonic calibration
    # ------------------------------------------------------------------------

    calibrated_probabilities = (
        calibrator.predict(
            raw_probabilities
        )
    )

    calibrated_probabilities = (
        np.asarray(
            calibrated_probabilities,
            dtype=np.float32,
        )
    )

    # Numerical safety.
    calibrated_probabilities = np.clip(
        calibrated_probabilities,
        0.0,
        1.0,
    )

    if not np.isfinite(
        calibrated_probabilities
    ).all():
        raise ValueError(
            "Calibrator produced NaN/Inf probabilities."
        )

    return calibrated_probabilities


# ============================================================================
# BUILD TEST INDEXES
# ============================================================================

def build_all_test_indexes():
    """
    Build exactly the same families of indexes used by the training pipeline.
    """

    print("\n" + "=" * 80)
    print("BUILDING TEST BLOCKING INDEXES")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Token-sorted name
    # ------------------------------------------------------------------------

    print(
        "\n[1/8] Token-sorted name index - Source 2"
    )

    name_exact_s2 = (
        build_token_sorted_name_index(
            str(S2_FILE)
        )
    )

    print(
        "\n[2/8] Token-sorted name index - Source 3"
    )

    name_exact_s3 = (
        build_token_sorted_name_index(
            str(S3_FILE)
        )
    )

    # ------------------------------------------------------------------------
    # 2. Name token
    # ------------------------------------------------------------------------

    print(
        "\n[3/8] Name token index - Source 2"
    )

    name_token_s2 = (
        build_name_token_index(
            str(S2_FILE)
        )
    )

    print(
        "\n[4/8] Name token index - Source 3"
    )

    name_token_s3 = (
        build_name_token_index(
            str(S3_FILE)
        )
    )

    # ------------------------------------------------------------------------
    # 3. Address token
    # ------------------------------------------------------------------------

    print(
        "\n[5/8] Address token index - Source 2"
    )

    address_token_s2 = (
        build_address_token_index(
            str(S2_FILE)
        )
    )

    print(
        "\n[6/8] Address token index - Source 3"
    )

    address_token_s3 = (
        build_address_token_index(
            str(S3_FILE)
        )
    )

    # ------------------------------------------------------------------------
    # 4. Exact address
    # ------------------------------------------------------------------------

    print(
        "\n[7/8] Exact address index - Source 2"
    )

    exact_address_s2 = (
        build_exact_address_index(
            str(S2_FILE)
        )
    )

    print(
        "\n[8/8] Exact address index - Source 3"
    )

    exact_address_s3 = (
        build_exact_address_index(
            str(S3_FILE)
        )
    )

    return (
        name_exact_s2,
        name_exact_s3,
        name_token_s2,
        name_token_s3,
        address_token_s2,
        address_token_s3,
        exact_address_s2,
        exact_address_s3,
    )


# ============================================================================
# BUILD TARGET STORE
# ============================================================================

def build_target_store():
    """
    Load Source 2 + Source 3 feature records into one lookup dictionary.

    Candidate IDs from both sources are unique, so a combined store allows
    make_pair_features() to resolve either source transparently.
    """

    print("\n" + "=" * 80)
    print("LOADING TEST FEATURE STORES")
    print("=" * 80)

    print(
        "\nSource 2:"
    )

    source2_store = load_source_store(
        str(S2_FILE)
    )

    print(
        f"Source 2 records: "
        f"{len(source2_store):,}"
    )

    print(
        "\nSource 3:"
    )

    source3_store = load_source_store(
        str(S3_FILE)
    )

    print(
        f"Source 3 records: "
        f"{len(source3_store):,}"
    )

    combined = {}

    combined.update(
        source2_store
    )

    combined.update(
        source3_store
    )

    print(
        f"\nCombined target store: "
        f"{len(combined):,}"
    )

    del source2_store
    del source3_store

    gc.collect()

    return combined


# ============================================================================
# SMOKE TEST
# ============================================================================

def run_smoke_test(
    indexes,
    target_store,
    model,
    calibrator,
    threshold,
    top_k,
):
    """
    Run a few actual Source-1 rows through:

        blocking
        ->
        features
        ->
        LightGBM
        ->
        calibration
        ->
        threshold

    No final output files are written.
    """

    print("\n" + "=" * 80)
    print("RUNNING SMOKE TEST")
    print("=" * 80)

    checked = 0

    for chunk in source1_reader():

        for row in chunk.itertuples(
            index=False
        ):

            s1_id = safe_text(
                row.entity_id
            )

            country = safe_text(
                row.country
            )

            name = safe_text(
                row.business_name
            )

            address = safe_text(
                row.business_address
            )

            ranked = generate_candidates(
                country,
                name,
                address,
                indexes,
                top_k,
            )

            if not ranked:
                print(
                    f"S1={s1_id} -> no candidates"
                )

                checked += 1

                if checked >= 5:
                    break

                continue

            X = build_feature_matrix(
                country,
                name,
                address,
                ranked,
                target_store,
            )

            probabilities = predict_features(
                X,
                model,
                calibrator,
            )

            candidate_ids = [
                candidate_id
                for candidate_id, _score
                in ranked
            ]

            selected = [
                candidate_id
                for candidate_id, probability
                in zip(
                    candidate_ids,
                    probabilities,
                )
                if probability >= threshold
            ]

            print(
                f"S1={s1_id}"
                f" | candidates={len(candidate_ids)}"
                f" | feature_shape={X.shape}"
                f" | probability_min={probabilities.min():.5f}"
                f" | probability_max={probabilities.max():.5f}"
                f" | selected={len(selected)}"
            )

            checked += 1

            if checked >= 5:
                break

        if checked >= 5:
            break

    if checked == 0:
        raise RuntimeError(
            "Smoke test could not process any Source-1 entity."
        )

    print(
        "\nSMOKE TEST PASSED"
    )

    print(
        "The blocking, feature generation, LightGBM, "
        "and calibration interfaces are compatible."
    )

    print(
        "No final output files were modified."
    )


# ============================================================================
# LOCAL OUTPUT VALIDATION
# ============================================================================

def validate_outputs(
    expected_s1_rows: int,
) -> None:
    """
    Validate generated output files without loading them into pandas.

    Checks:
    - headers
    - row counts
    - unique Source-1 IDs
    - matching IDs = candidate IDs
    - final matches are a subset of candidates
    - no duplicate final match IDs
    """

    print("\n" + "=" * 80)
    print("LOCAL OUTPUT VALIDATION")
    print("=" * 80)

    if not MATCHING_FILE.exists():
        raise FileNotFoundError(
            f"Missing output file: {MATCHING_FILE}"
        )

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing output file: {CANDIDATE_FILE}"
        )

    matching_s1_ids = set()
    candidate_map = {}
    candidate_s1_ids = set()

    matching_rows = 0
    candidate_rows = 0

    # ------------------------------------------------------------------------
    # Candidate file
    # ------------------------------------------------------------------------

    with open(
        CANDIDATE_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as candidate_fp:

        reader = csv.DictReader(
            candidate_fp,
            delimiter="\t",
        )

        expected_header = [
            "source1_entity_id",
            "candidate_entity_ids",
        ]

        if reader.fieldnames != expected_header:
            raise ValueError(
                "\nWrong candidate_pairs.tsv header.\n"
                f"Expected: {expected_header}\n"
                f"Received: {reader.fieldnames}"
            )

        for row in reader:

            s1_id = row[
                "source1_entity_id"
            ]

            if s1_id in candidate_s1_ids:
                raise ValueError(
                    f"Duplicate S1 ID in candidate_pairs.tsv: {s1_id}"
                )

            candidate_s1_ids.add(
                s1_id
            )

            raw_candidates = row[
                "candidate_entity_ids"
            ]

            candidates = {
                candidate
                for candidate
                in raw_candidates.split(",")
                if candidate
            }

            candidate_map[
                s1_id
            ] = candidates

            candidate_rows += 1

    # ------------------------------------------------------------------------
    # Matching file
    # ------------------------------------------------------------------------

    with open(
        MATCHING_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as matching_fp:

        reader = csv.DictReader(
            matching_fp,
            delimiter="\t",
        )

        expected_header = [
            "source1_entity_id",
            "matched_entity_ids",
        ]

        if reader.fieldnames != expected_header:
            raise ValueError(
                "\nWrong matching_results.tsv header.\n"
                f"Expected: {expected_header}\n"
                f"Received: {reader.fieldnames}"
            )

        for row in reader:

            s1_id = row[
                "source1_entity_id"
            ]

            if s1_id in matching_s1_ids:
                raise ValueError(
                    f"Duplicate S1 ID in matching_results.tsv: {s1_id}"
                )

            matching_s1_ids.add(
                s1_id
            )

            if s1_id not in candidate_map:
                raise ValueError(
                    f"S1 {s1_id} exists in matching output "
                    "but not candidate output."
                )

            matched = [
                candidate
                for candidate
                in row[
                    "matched_entity_ids"
                ].split(",")
                if candidate
            ]

            # No duplicate matched IDs.
            if len(matched) != len(
                set(matched)
            ):
                raise ValueError(
                    f"Duplicate matched IDs for S1 {s1_id}."
                )

            # Every match must belong to the final candidate set.
            candidates = candidate_map[
                s1_id
            ]

            outside_candidate_set = [
                candidate
                for candidate
                in matched
                if candidate
                not in candidates
            ]

            if outside_candidate_set:
                raise ValueError(
                    f"\nS1 {s1_id} has final matches that are "
                    "not present in candidate_pairs.tsv:\n"
                    f"{outside_candidate_set[:10]}"
                )

            matching_rows += 1

    # ------------------------------------------------------------------------
    # Global checks
    # ------------------------------------------------------------------------

    if matching_rows != expected_s1_rows:
        raise ValueError(
            "\nWrong number of matching rows.\n"
            f"Expected: {expected_s1_rows:,}\n"
            f"Received: {matching_rows:,}"
        )

    if candidate_rows != expected_s1_rows:
        raise ValueError(
            "\nWrong number of candidate rows.\n"
            f"Expected: {expected_s1_rows:,}\n"
            f"Received: {candidate_rows:,}"
        )

    if matching_s1_ids != candidate_s1_ids:
        missing_from_matching = (
            candidate_s1_ids
            - matching_s1_ids
        )

        missing_from_candidates = (
            matching_s1_ids
            - candidate_s1_ids
        )

        raise ValueError(
            "\nSource-1 ID sets differ between output files.\n"
            f"Missing from matching: {list(missing_from_matching)[:5]}\n"
            f"Missing from candidates: {list(missing_from_candidates)[:5]}"
        )

    print(
        "Headers          : PASS"
    )

    print(
        "Row counts       : PASS"
    )

    print(
        "Unique S1 IDs    : PASS"
    )

    print(
        "Match subset rule: PASS"
    )


# ============================================================================
# FULL INFERENCE
# ============================================================================

def run_full_inference(
    indexes,
    target_store,
    model,
    calibrator,
    threshold,
    top_k,
) -> None:

    print("\n" + "=" * 80)
    print("STARTING FULL TEST INFERENCE")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Clean only temporary files.
    #
    # Existing final outputs are preserved until the new run completes.
    # ------------------------------------------------------------------------

    safe_remove(
        TEMP_MATCHING_FILE
    )

    safe_remove(
        TEMP_CANDIDATE_FILE
    )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    total_s1 = 0
    total_candidates = 0
    total_matches = 0

    zero_candidate_s1 = 0
    zero_match_s1 = 0

    source2_matches = 0
    source3_matches = 0

    candidate_histogram = Counter()
    match_histogram = Counter()

    # ------------------------------------------------------------------------
    # Open temporary output files.
    # ------------------------------------------------------------------------

    with open(
        TEMP_MATCHING_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as matching_fp, open(
        TEMP_CANDIDATE_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as candidate_fp:

        matching_writer = csv.writer(
            matching_fp,
            delimiter="\t",
            lineterminator="\n",
        )

        candidate_writer = csv.writer(
            candidate_fp,
            delimiter="\t",
            lineterminator="\n",
        )

        # Official headers.
        matching_writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        candidate_writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

        # ====================================================================
        # STREAM SOURCE 1
        # ====================================================================

        for chunk in source1_reader():

            # Store the matrices for the current bounded chunk.
            batch_feature_matrices = []

            batch_s1_ids = []

            batch_candidate_ids = []

            # ----------------------------------------------------------------
            # BLOCKING + FEATURES
            # ----------------------------------------------------------------

            for row in chunk.itertuples(
                index=False
            ):

                s1_id = safe_text(
                    row.entity_id
                )

                country = safe_text(
                    row.country
                )

                name = safe_text(
                    row.business_name
                )

                address = safe_text(
                    row.business_address
                )

                if not s1_id:
                    raise ValueError(
                        "Encountered an empty Source-1 entity_id."
                    )

                # ------------------------------------------------------------
                # Blocking
                # ------------------------------------------------------------

                ranked_candidates = (
                    generate_candidates(
                        country,
                        name,
                        address,
                        indexes,
                        top_k,
                    )
                )

                candidate_ids = [
                    candidate_id
                    for candidate_id, _score
                    in ranked_candidates
                ]

                # ------------------------------------------------------------
                # IMPORTANT:
                #
                # Write candidate_pairs.tsv BEFORE ML.
                # ------------------------------------------------------------

                candidate_writer.writerow(
                    [
                        s1_id,
                        ",".join(
                            candidate_ids
                        ),
                    ]
                )

                candidate_count = len(
                    candidate_ids
                )

                total_candidates += (
                    candidate_count
                )

                candidate_histogram[
                    candidate_count
                ] += 1

                if candidate_count == 0:
                    zero_candidate_s1 += 1

                # ------------------------------------------------------------
                # Features
                # ------------------------------------------------------------

                X = build_feature_matrix(
                    country,
                    name,
                    address,
                    ranked_candidates,
                    target_store,
                )

                batch_feature_matrices.append(
                    X
                )

                batch_s1_ids.append(
                    s1_id
                )

                batch_candidate_ids.append(
                    candidate_ids
                )

            # ----------------------------------------------------------------
            # ONE BATCHED MODEL CALL
            # ----------------------------------------------------------------

            non_empty_matrices = [
                X
                for X in batch_feature_matrices
                if X.size > 0
            ]

            if non_empty_matrices:

                combined_X = np.vstack(
                    non_empty_matrices
                ).astype(
                    np.float32,
                    copy=False,
                )

                probabilities = (
                    predict_features(
                        combined_X,
                        model,
                        calibrator,
                    )
                )

            else:

                probabilities = np.empty(
                    0,
                    dtype=np.float32,
                )

            # ----------------------------------------------------------------
            # DECODE EACH SOURCE-1 ENTITY
            # ----------------------------------------------------------------

            probability_offset = 0

            for (
                s1_id,
                candidate_ids,
                X,
            ) in zip(
                batch_s1_ids,
                batch_candidate_ids,
                batch_feature_matrices,
            ):

                candidate_count = len(
                    candidate_ids
                )

                if candidate_count == 0:

                    matched_ids = []

                else:

                    start = (
                        probability_offset
                    )

                    end = (
                        start
                        + candidate_count
                    )

                    entity_probabilities = (
                        probabilities[
                            start:end
                        ]
                    )

                    probability_offset = end

                    # --------------------------------------------------------
                    # Threshold
                    # --------------------------------------------------------

                    selected = []

                    for index, candidate_id in enumerate(
                        candidate_ids
                    ):

                        probability = float(
                            entity_probabilities[
                                index
                            ]
                        )

                        if probability >= threshold:
                            selected.append(
                                (
                                    candidate_id,
                                    probability,
                                )
                            )

                    # Strongest candidates first.
                    selected.sort(
                        key=lambda pair: (
                            -pair[1],
                            pair[0],
                        )
                    )

                    # Defensive de-duplication.
                    seen = set()

                    matched_ids = []

                    for (
                        candidate_id,
                        _probability,
                    ) in selected:

                        if candidate_id in seen:
                            continue

                        seen.add(
                            candidate_id
                        )

                        matched_ids.append(
                            candidate_id
                        )

                # ------------------------------------------------------------
                # One output row for EVERY S1.
                # ------------------------------------------------------------

                matching_writer.writerow(
                    [
                        s1_id,
                        ",".join(
                            matched_ids
                        ),
                    ]
                )

                total_s1 += 1

                match_count = len(
                    matched_ids
                )

                total_matches += (
                    match_count
                )

                match_histogram[
                    match_count
                ] += 1

                if match_count == 0:
                    zero_match_s1 += 1

                for candidate_id in matched_ids:

                    if candidate_id.startswith(
                        "S3-"
                    ):
                        source3_matches += 1

                    else:
                        source2_matches += 1

            # ----------------------------------------------------------------
            # Probability alignment safety check.
            # ----------------------------------------------------------------

            if probability_offset != len(
                probabilities
            ):
                raise RuntimeError(
                    "\nPrediction alignment failure.\n"
                    f"Consumed: {probability_offset}\n"
                    f"Produced: {len(probabilities)}"
                )

            # ----------------------------------------------------------------
            # Flush files
            # ----------------------------------------------------------------

            matching_fp.flush()
            candidate_fp.flush()

            # ----------------------------------------------------------------
            # Progress
            # ----------------------------------------------------------------

            if (
                total_s1 > 0
                and total_s1 % PROGRESS_EVERY == 0
            ):

                average_candidates = (
                    total_candidates
                    / total_s1
                )

                average_matches = (
                    total_matches
                    / total_s1
                )

                print(
                    "\nPROGRESS"
                    f" | S1={total_s1:,}"
                    f" | candidates={total_candidates:,}"
                    f" | avg_candidates={average_candidates:.2f}"
                    f" | links={total_matches:,}"
                    f" | avg_links={average_matches:.3f}"
                    f" | zero_candidates={zero_candidate_s1:,}"
                    f" | zero_matches={zero_match_s1:,}"
                )

            # ----------------------------------------------------------------
            # Memory release
            # ----------------------------------------------------------------

            del (
                batch_feature_matrices,
                batch_s1_ids,
                batch_candidate_ids,
                non_empty_matrices,
                probabilities,
                chunk,
            )

            gc.collect()

    # =========================================================================
    # ATOMIC FINALIZATION
    # =========================================================================

    # Only now replace the final outputs.
    safe_remove(
        MATCHING_FILE
    )

    safe_remove(
        CANDIDATE_FILE
    )

    TEMP_MATCHING_FILE.replace(
        MATCHING_FILE
    )

    TEMP_CANDIDATE_FILE.replace(
        CANDIDATE_FILE
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    average_candidates = (
        total_candidates / total_s1
        if total_s1
        else 0.0
    )

    average_matches = (
        total_matches / total_s1
        if total_s1
        else 0.0
    )

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)

    print(
        f"\nS1 entities processed : "
        f"{total_s1:,}"
    )

    print(
        f"Total candidate pairs : "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates/S1 : "
        f"{average_candidates:.3f}"
    )

    print(
        f"Total predicted links : "
        f"{total_matches:,}"
    )

    print(
        f"Average links/S1      : "
        f"{average_matches:.3f}"
    )

    print(
        f"Zero-candidate S1     : "
        f"{zero_candidate_s1:,}"
    )

    print(
        f"Zero-match S1         : "
        f"{zero_match_s1:,}"
    )

    print(
        f"S2 predicted links    : "
        f"{source2_matches:,}"
    )

    print(
        f"S3 predicted links    : "
        f"{source3_matches:,}"
    )

    print(
        f"\nThreshold             : "
        f"{threshold:.6f}"
    )

    print(
        f"Candidate Top-K       : "
        f"{top_k}"
    )

    # ------------------------------------------------------------------------
    # Match distribution
    # ------------------------------------------------------------------------

    print(
        "\nMatch-count distribution:"
    )

    for count in sorted(
        match_histogram
    ):
        print(
            f"  {count:>2} matches : "
            f"{match_histogram[count]:,}"
        )

    print(
        "\nOutput files:"
    )

    print(
        f"  {MATCHING_FILE}"
    )

    print(
        f"  {CANDIDATE_FILE}"
    )

    # ------------------------------------------------------------------------
    # Local validation
    # ------------------------------------------------------------------------

    validate_outputs(
        expected_s1_rows=total_s1
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_arguments()

    print("=" * 80)
    print(
        "COMPETITIVE LIGHTGBM ENTITY RESOLUTION"
    )
    print(
        "PRODUCTION TEST INFERENCE"
    )
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------------

    check_required_files()

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    (
        model,
        calibrator,
        threshold,
        top_k,
    ) = load_model_artifact()

    # ------------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------------

    indexes = build_all_test_indexes()

    # ------------------------------------------------------------------------
    # Target store
    # ------------------------------------------------------------------------

    target_store = build_target_store()

    print(
        f"\nTarget records available: "
        f"{len(target_store):,}"
    )

    # ------------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------------

    if args.smoke_test:

        run_smoke_test(
            indexes,
            target_store,
            model,
            calibrator,
            threshold,
            top_k,
        )

        return

    # ------------------------------------------------------------------------
    # Full inference
    # ------------------------------------------------------------------------

    run_full_inference(
        indexes,
        target_store,
        model,
        calibrator,
        threshold,
        top_k,
    )

    print("\n" + "=" * 80)
    print("READY FOR OFFICIAL VALIDATION")
    print("=" * 80)

    print(
        "\nRun the official validator:"
    )

    print(
        "python .\\utils\\validate_submission.py "
        "--matching .\\output\\matching_results.tsv "
        "--candidate .\\output\\candidate_pairs.tsv "
        "--test-dir .\\dataset\\test"
    )

    print(
        "\nThen run the ID-existence check:"
    )

    print(
        "python .\\utils\\validate_submission.py "
        "--matching .\\output\\matching_results.tsv "
        "--candidate .\\output\\candidate_pairs.tsv "
        "--test-dir .\\dataset\\test "
        "--check-ids"
    )

    print("\nDo not submit until both checks pass.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()