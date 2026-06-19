#!/usr/bin/env python3
"""
Build and save TF-IDF vectorizers for trajectory retrieval corpus.

This script:
1. Loads all trajectories from LanceDB for each environment
2. Builds TF-IDF vectorizers on different text components:
   - Full text (concatenated goal, state, context, progress)
   - Goal only
   - State only
   - Context only
3. Saves fitted vectorizers to disk for later use in enrichment

Run this once to prepare TF-IDF corpus for all environments.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import lancedb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Environment configurations (matching enrich_disagreements.py)
ENVIRONMENT_CONFIGS = {
    "alfworld": {
        "base_path": os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "alfworld"),
        "lancedb_path": os.path.join(
            os.environ.get("MATM_DATA_ROOT", "environments"),
            "train_only_lancedb/alfworld/lancedb_indices",
        ),
        "table_name": "alfworld",
        "output_dir": "alfworld",
    },
    "webarena": {
        "base_path": os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena"),
        "lancedb_path": os.path.join(
            os.environ.get("MATM_DATA_ROOT", "environments"),
            "train_only_lancedb/webarena/lancedb_indices",
        ),
        "table_name": "webarena",
        "output_dir": "webarena",
    },
}

# TF-IDF parameters
TFIDF_PARAMS = {
    "max_features": None,  # No limit on vocabulary size
    "min_df": 1,  # Minimum document frequency
    "max_df": 0.95,  # Ignore terms that appear in >95% of documents
    "ngram_range": (1, 1),  # Unigrams only for TF-IDF
    "lowercase": True,
    "strip_accents": "unicode",
    "token_pattern": r"(?u)\b\w+\b",  # Word tokens
}


def load_trajectories_from_lancedb(lancedb_path: str, table_name: str) -> pd.DataFrame:
    """
    Load all trajectory data from LanceDB table.

    Args:
        lancedb_path: Path to LanceDB directory
        table_name: Name of table to load

    Returns:
        DataFrame with all trajectory rows
    """
    print(f"  Connecting to LanceDB: {lancedb_path}")
    db = lancedb.connect(lancedb_path)
    table = db.open_table(table_name)

    print(f"  Loading all rows from table: {table_name}")
    # Load all data - this returns a PyArrow table
    df = table.to_pandas()

    print(f"  ✓ Loaded {len(df):,} trajectories")
    return df


def extract_text_components(row: pd.Series) -> Dict[str, str]:
    """
    Extract text components from a trajectory row.

    Args:
        row: DataFrame row with trajectory data

    Returns:
        Dictionary with text components
    """
    goal = row.get("key_raw_goal", "") or ""
    state = row.get("key_raw_state", "") or ""
    context = row.get("key_raw_context", "") or ""
    progress = row.get("key_raw_progress", "") or ""

    # Clean up whitespace
    goal = goal.strip()
    state = state.strip()
    context = context.strip()
    progress = progress.strip()

    # Full text: concatenate all components with labels (matching query format)
    full_text = (
        f"goal: {goal} | state: {state} | context: {context} | progress: {progress}"
    )

    return {
        "full": full_text,
        "goal": goal,
        "state": state,
        "context": context,
    }


def build_corpus_texts(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build corpus texts for each component from all trajectories.

    Args:
        df: DataFrame with all trajectory rows

    Returns:
        Dictionary mapping component name to list of texts
    """
    print(f"  Extracting text components from {len(df):,} trajectories...")

    corpora = {
        "full": [],
        "goal": [],
        "state": [],
        "context": [],
    }

    # Process rows with progress bar
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="  Extracting texts", unit="traj"
    ):
        components = extract_text_components(row)

        for component_name, text in components.items():
            corpora[component_name].append(text)

    # Print statistics
    for component_name, texts in corpora.items():
        non_empty = sum(1 for t in texts if t)
        print(
            f"    - {component_name}: {non_empty:,} non-empty texts / {len(texts):,} total"
        )

    return corpora


def build_and_save_vectorizer(
    corpus: List[str], component_name: str, output_path: Path, params: Dict
) -> TfidfVectorizer:
    """
    Build TF-IDF vectorizer and save to disk.

    Args:
        corpus: List of text documents
        component_name: Name of component (for logging)
        output_path: Path to save vectorizer
        params: TF-IDF parameters

    Returns:
        Fitted TfidfVectorizer
    """
    print(f"    Building TF-IDF vectorizer for: {component_name}")

    # Filter out empty strings for fitting
    non_empty_corpus = [text for text in corpus if text.strip()]

    if not non_empty_corpus:
        print(f"      ⚠️  No non-empty texts for {component_name}, skipping")
        return None

    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(**params)
    vectorizer.fit(non_empty_corpus)

    # Print statistics
    vocab_size = len(vectorizer.vocabulary_)
    print(f"      ✓ Vocabulary size: {vocab_size:,} terms")

    # Save to disk
    with open(output_path, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"      ✓ Saved to: {output_path}")

    return vectorizer


def process_environment(env_name: str):
    """
    Process a single environment: load data, build vectorizers, save.

    Args:
        env_name: Name of environment to process
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING ENVIRONMENT: {env_name.upper()}")
    print(f"{'='*80}\n")

    config = ENVIRONMENT_CONFIGS[env_name]

    # Setup output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / config["output_dir"]
    output_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Load all trajectories from LanceDB
        df = load_trajectories_from_lancedb(
            lancedb_path=config["lancedb_path"], table_name=config["table_name"]
        )

        # Step 2: Extract text components
        corpora = build_corpus_texts(df)

        # Step 3: Build and save TF-IDF vectorizers for each component
        print(f"\n  Building TF-IDF vectorizers...")

        vectorizers = {}
        for component_name, corpus in corpora.items():
            output_path = output_dir / f"tfidf_vectorizer_{component_name}.pkl"
            vectorizer = build_and_save_vectorizer(
                corpus=corpus,
                component_name=component_name,
                output_path=output_path,
                params=TFIDF_PARAMS,
            )
            vectorizers[component_name] = vectorizer

        # Step 4: Save metadata about the corpus
        metadata = {
            "environment": env_name,
            "num_trajectories": len(df),
            "tfidf_params": TFIDF_PARAMS,
            "components": {},
        }

        for component_name, corpus in corpora.items():
            if vectorizers[component_name] is not None:
                metadata["components"][component_name] = {
                    "num_documents": len(corpus),
                    "num_non_empty": sum(1 for t in corpus if t.strip()),
                    "vocabulary_size": len(vectorizers[component_name].vocabulary_),
                }

        metadata_path = output_dir / "tfidf_corpus_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  ✓ Saved metadata to: {metadata_path}")

        print(f"\n{'='*80}")
        print(f"✅ COMPLETED: {env_name.upper()}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR processing {env_name.upper()}: {e}")
        print(f"{'='*80}\n")
        raise


def main():
    """Main function to process all environments."""
    print(f"\n{'='*80}")
    print(f"TF-IDF CORPUS BUILDER")
    print(f"{'='*80}")
    print(f"Building TF-IDF vectorizers for trajectory retrieval")
    print(f"Environments: {', '.join(ENVIRONMENT_CONFIGS.keys())}")
    print(f"{'='*80}\n")

    # Process each environment sequentially
    # (Could parallelize across environments, but each is memory-intensive)
    for env_name in ENVIRONMENT_CONFIGS.keys():
        process_environment(env_name)

    print(f"\n{'='*80}")
    print(f"✅ ALL ENVIRONMENTS COMPLETED")
    print(f"{'='*80}\n")

    # Print summary
    print("Summary of output files:")
    script_dir = Path(__file__).parent
    for env_name, config in ENVIRONMENT_CONFIGS.items():
        output_dir = script_dir / config["output_dir"]
        print(f"\n{env_name.upper()}:")
        print(f"  Location: {output_dir}")

        # List generated files
        pkl_files = list(output_dir.glob("tfidf_vectorizer_*.pkl"))
        for pkl_file in sorted(pkl_files):
            size_mb = pkl_file.stat().st_size / (1024 * 1024)
            print(f"    - {pkl_file.name} ({size_mb:.2f} MB)")

        metadata_file = output_dir / "tfidf_corpus_metadata.json"
        if metadata_file.exists():
            print(f"    - {metadata_file.name}")

    print(f"\n{'='*80}")
    print("These vectorizers can now be loaded in enrich_disagreements.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
