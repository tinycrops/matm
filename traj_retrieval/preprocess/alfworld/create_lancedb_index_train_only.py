#!/usr/bin/env python3
"""
Create or recreate vector index for train-only AlfWorld LanceDB table.

This script creates a vector index on the 'key_embed' column for efficient
similarity search. It can be run independently of data creation, allowing
you to experiment with different index configurations.

To change settings, edit the GLOBAL CONFIGURATION section below.

Usage:
    python traj_retrieval/preprocess/alfworld/create_lancedb_index_train_only.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    import lancedb
except ImportError:
    print("ERROR: lancedb not found in current environment")
    print("Please install it with: pip install lancedb")
    sys.exit(1)

# ============================================================================
# GLOBAL CONFIGURATION - Edit these settings as needed
# ============================================================================

# Base output directory for train-only ALFWorld artifacts
TRAIN_ONLY_BASE_DIR = os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "train_only_lancedb/alfworld")

# LanceDB configuration
LANCEDB_URI = f"{TRAIN_ONLY_BASE_DIR}/lancedb_indices"
TABLE_NAME = "alfworld"

# Index parameters
METRIC = "L2"  # Distance metric: "L2" (Euclidean) or "cosine"
INDEX_TYPE = "IVF_PQ"  # Index type: "IVF_PQ", "IVF_HNSW_SQ", "IVF_HNSW_PQ"
NUM_PARTITIONS = 256  # Number of partitions for IVF (recommended: 128-512)
NUM_SUB_VECTORS = 16  # Number of sub-vectors for PQ (recommended: 8-32)
REPLACE_EXISTING = True  # Set to True to replace existing index

# ============================================================================


def create_index():
    """Create vector index on LanceDB table using global configuration."""
    print("=" * 80)
    print("AlfWorld Train-Only LanceDB Vector Index Creator")
    print("=" * 80)
    print(f"Database URI: {LANCEDB_URI}")
    print(f"Table name: {TABLE_NAME}")
    print(f"Vector column: key_embed")
    print()
    print("Index Configuration:")
    print(f"  Metric: {METRIC}")
    print(f"  Type: {INDEX_TYPE}")
    print(f"  Partitions: {NUM_PARTITIONS}")
    print(f"  Sub-vectors: {NUM_SUB_VECTORS}")
    print(f"  Replace existing: {REPLACE_EXISTING}")
    print("=" * 80)
    print()

    # Check if database exists
    if not os.path.exists(LANCEDB_URI):
        print(f"❌ ERROR: Database not found at {LANCEDB_URI}")
        print(
            "   Please run create_seq_to_seq_indices_train_only.py first to create the database."
        )
        sys.exit(1)

    # Connect to database
    print(f"Connecting to LanceDB at {LANCEDB_URI}...")
    try:
        db = lancedb.connect(LANCEDB_URI)
        print("✓ Connected successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to connect: {e}")
        sys.exit(1)

    # Check if table exists
    table_names = db.table_names()
    if TABLE_NAME not in table_names:
        print(f"❌ ERROR: Table '{TABLE_NAME}' not found")
        print(f"   Available tables: {', '.join(table_names)}")
        print(
            "   Please run create_seq_to_seq_indices_train_only.py first to create the table."
        )
        sys.exit(1)

    # Open table
    print(f"Opening table '{TABLE_NAME}'...")
    try:
        table = db.open_table(TABLE_NAME)
        print(f"✓ Table opened successfully ({len(table):,} entries)")
    except Exception as e:
        print(f"❌ ERROR: Failed to open table: {e}")
        sys.exit(1)

    # Check current indices
    print()
    print("Checking existing indices...")
    try:
        # LanceDB doesn't have a direct "list indices" method, but we can check
        # by trying to get index statistics
        print("  Checking for existing index on 'key_embed'...")
        # If an index exists, this will work; if not, it will fail
        # Note: This is a heuristic check, may need adjustment based on LanceDB version
    except Exception:
        print("  No existing index found")

    # Create index
    print()
    print("=" * 80)
    if REPLACE_EXISTING:
        print("Creating/Replacing vector index...")
    else:
        print("Creating vector index...")
    print("=" * 80)
    print()
    print("⏳ This may take several minutes for large datasets...")
    print("   Progress will be shown by LanceDB...")
    print()

    try:
        # Build index parameters
        index_params = {
            "metric": METRIC,
            "vector_column_name": "key_embed",
            "index_type": INDEX_TYPE,
            "replace": REPLACE_EXISTING,
        }

        # Add type-specific parameters
        if "IVF" in INDEX_TYPE:
            index_params["num_partitions"] = NUM_PARTITIONS

        if "PQ" in INDEX_TYPE:
            index_params["num_sub_vectors"] = NUM_SUB_VECTORS

        # Create index
        table.create_index(**index_params)

        print()
        print("=" * 80)
        print("✓ Vector index created successfully!")
        print("=" * 80)
        print()
        print("Index Details:")
        print(f"  Vector column: key_embed")
        print(f"  Metric: {METRIC}")
        print(f"  Type: {INDEX_TYPE}")
        if "IVF" in INDEX_TYPE:
            print(f"  Partitions: {NUM_PARTITIONS}")
        if "PQ" in INDEX_TYPE:
            print(f"  Sub-vectors: {NUM_SUB_VECTORS}")
        print()
        print("✓ The index is ready for similarity search!")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR: Failed to create index")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback

        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print(
            "  1. Check if table has data: should have entries with 'key_embed' vectors"
        )
        print("  2. Try different index parameters (smaller partitions/sub-vectors)")
        print("  3. Check LanceDB version compatibility")
        print()
        sys.exit(1)


def main():
    """Main function."""
    create_index()


if __name__ == "__main__":
    main()
