#!/usr/bin/env python3
"""
LanceDB client for trajectory retrieval.

This client provides a clean interface for searching LanceDB tables
with trajectory data.
"""

import os
from typing import Optional, Dict, Any, List, Tuple
import json

try:
    import lancedb
except ImportError:
    lancedb = None

from sentence_transformers import SentenceTransformer
from .search_config_parser import build_lancedb_filter, merge_search_config


class LanceDBClient:
    """
    Client for interacting with LanceDB trajectory database.

    This class handles:
    - Connection to LanceDB
    - Embedding queries
    - Searching for similar trajectories
    - Formatting search results
    """

    def __init__(
        self,
        db_uri: str,
        model_name: str = "intfloat/e5-base",
        table_name: str = "alfworld",
        search_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LanceDB client.

        Args:
            db_uri: Path to LanceDB database directory
            model_name: SentenceTransformer model name for encoding
            table_name: Name of the table to search in
            search_config: Optional search configuration (filters, top_k, etc.)
        """
        if lancedb is None:
            raise ImportError(
                "lancedb not found. Please install it with: pip install lancedb"
            )

        self.db_uri = db_uri
        self.model_name = model_name
        self.table_name = table_name
        self._model = None  # Lazy loading
        self._db = None  # Lazy loading
        self._table = None  # Lazy loading

        # Resolve defaults once so search() can read one normalized config shape.
        self.search_config = merge_search_config(search_config)

        print(f"[LanceDBClient] Initialized")
        print(f"[LanceDBClient]   Database URI: {db_uri}")
        print(f"[LanceDBClient]   Model: {model_name}")
        print(f"[LanceDBClient]   Table: {table_name}")

    @property
    def model(self):
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            import torch
            import signal
            import platform

            device_override = (
                str(os.environ.get("LANCEDB_EMBED_DEVICE", "") or "").strip().lower()
            )
            if device_override:
                if device_override not in {"cpu", "cuda", "mps"}:
                    raise ValueError(
                        "Invalid LANCEDB_EMBED_DEVICE. Expected one of: cpu, cuda, mps."
                    )
                device = device_override
                print(
                    f"[LanceDBClient] Using device override from LANCEDB_EMBED_DEVICE={device}"
                )
            else:
                # Auto-detect best available device
                if torch.cuda.is_available():
                    device = "cuda"
                    print(f"[LanceDBClient] CUDA detected, using GPU")
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    device = "cpu"  # Force CPU on macOS for stability
                    print(
                        f"[LanceDBClient] MPS available but using CPU for stability on macOS"
                    )
                else:
                    device = "cpu"
                    print(f"[LanceDBClient] Using CPU (no GPU detected)")

            timeout_seconds_raw = str(
                os.environ.get("LANCEDB_MODEL_LOAD_TIMEOUT_S", "60") or "60"
            ).strip()
            try:
                timeout_seconds = int(timeout_seconds_raw)
            except ValueError as error:
                raise ValueError(
                    "LANCEDB_MODEL_LOAD_TIMEOUT_S must be a positive integer."
                ) from error
            if timeout_seconds <= 0:
                raise ValueError(
                    "LANCEDB_MODEL_LOAD_TIMEOUT_S must be a positive integer."
                )
            print(f"[LanceDBClient] Model load timeout: {timeout_seconds}s")

            # Use timeout to prevent hanging (only on Unix-like systems)
            class ModelLoadTimeout(Exception):
                pass

            def timeout_handler(signum, frame):
                raise ModelLoadTimeout(
                    f"Model loading timed out after {timeout_seconds} seconds"
                )

            timeout_supported = platform.system() != "Windows"
            old_handler = None

            if timeout_supported:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            try:
                print(
                    f"[LanceDBClient] Loading SentenceTransformer model '{self.model_name}' on {device}..."
                )
                self._model = SentenceTransformer(self.model_name, device=device)

                if timeout_supported:
                    signal.alarm(0)

                print(
                    f"[LanceDBClient] ✅ Model loaded successfully on device: {device}"
                )
            except ModelLoadTimeout as e:
                if timeout_supported:
                    signal.alarm(0)
                print(f"[LanceDBClient] ❌ {e}")
                raise RuntimeError("Model loading timed out. Check your environment.")
            except Exception as e:
                if timeout_supported and signal.alarm:
                    signal.alarm(0)
                print(f"[LanceDBClient] ❌ Error loading model: {e}")
                raise
            finally:
                if timeout_supported and old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        return self._model

    @property
    def db(self):
        """Lazy load LanceDB connection."""
        if self._db is None:
            if not os.path.exists(self.db_uri):
                raise FileNotFoundError(
                    f"LanceDB directory not found: {self.db_uri}\n"
                    f"Please create indices first using create_lancedb_indices.py"
                )

            print(f"[LanceDBClient] Connecting to LanceDB at {self.db_uri}...")
            self._db = lancedb.connect(self.db_uri)
            print(f"[LanceDBClient] ✅ Connected successfully")

        return self._db

    @property
    def table(self):
        """Lazy load table."""
        if self._table is None:
            table_names = self.db.table_names()
            if self.table_name not in table_names:
                raise ValueError(
                    f"Table '{self.table_name}' not found in database.\n"
                    f"Available tables: {table_names}\n"
                    f"Please create the table first using create_lancedb_indices.py"
                )

            print(f"[LanceDBClient] Opening table '{self.table_name}'...")
            self._table = self.db.open_table(self.table_name)
            print(f"[LanceDBClient] ✅ Table opened successfully")

        return self._table

    def search(
        self,
        query_text: str,
        k: Optional[int] = None,
        filter_condition: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search LanceDB for similar trajectories using config-based filters.

        Args:
            query_text: Query text to embed and search
            k: Number of results to return (overrides config if provided)
            filter_condition: Optional SQL-like filter condition (overrides config if provided)
            context: Runtime context for dynamic filter values

        Returns:
            List of matching entries (dictionaries)
        """
        # Get search parameters from config or arguments
        candidate_config = self.search_config.get("candidate_generation", {})

        # Use k from argument, or fall back to config, or default to 1
        k = k if k is not None else candidate_config.get("top_k", 1)

        # Get vector field from config (default: key_embed)
        vector_field = candidate_config.get("query_embedding_field", "key_embed")

        # Build filter from config if not explicitly provided
        filter_debug_info = None
        if filter_condition is None and context is not None:
            # `query_text` drives vector similarity; `context` is only for resolving
            # dynamic filter placeholders like task/variation constraints.
            filters_config = candidate_config.get("filters", {})
            filter_condition, filter_debug_info = build_lancedb_filter(
                filters_config, context
            )

        # Log filter query for debugging
        print(f"\n{'─'*80}")
        print(f"🔍 LANCEDB QUERY CONSTRUCTION")
        print(f"{'─'*80}")
        if filter_debug_info:
            print(
                f"Context: task_name={context.get('task_name')}, variation_idx={context.get('variation_idx')}"
            )
            print(
                f"Base Filters: {len(filter_debug_info.get('base_conditions', []))} conditions"
            )
            for cond in filter_debug_info.get("base_conditions", []):
                print(f"  - {cond['field']} = {cond['value']}")
            if filter_debug_info.get("diversity_strategy"):
                print(f"Diversity Strategy: {filter_debug_info['diversity_strategy']}")
                print(
                    f"  Generated: {filter_debug_info.get('diversity_condition', 'N/A')}"
                )
            print(f"Final SQL WHERE Clause:")
            print(
                f"  {filter_condition if filter_condition else 'No filters (all trajectories)'}"
            )
        else:
            if filter_condition:
                print(f"SQL WHERE Clause: {filter_condition}")
            else:
                print(f"No filters applied")
        print(f"Vector Field: {vector_field}")
        print(f"Top K: {k}")
        print(f"{'─'*80}\n")

        # Convert the runtime query string into the same embedding space used by
        # the indexed `vector_field` column.
        query_embedding = self.model.encode(query_text).tolist()

        # LanceDB returns a query builder here; filters/limit are applied before
        # materializing the final rows into Python dictionaries.
        search_query = self.table.search(
            query_embedding, vector_column_name=vector_field
        )

        # Apply filter if provided
        if filter_condition:
            search_query = search_query.where(filter_condition)

        # Keep the resolved filter trace so retrieval metadata can log exactly
        # which constraints were active for this search call.
        self._last_filter_debug = filter_debug_info

        # Limit results
        search_query = search_query.limit(k)

        # Execute and convert to list of dicts
        results = search_query.to_pandas()

        if results.empty:
            return []

        # Downstream reranking/formatting expects raw LanceDB rows as plain dicts.
        return results.to_dict("records")

    def check_candidates_exist(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[str]]:
        """
        Check if at least one candidate exists with the given filter constraints.

        This is a lightweight preflight check that runs before episode execution.
        It verifies that retrieval will be possible given the configured filters.

        Args:
            context: Runtime context for dynamic filter values (task_name, variation_idx, etc.)

        Returns:
            Tuple of (candidates_exist, count, filter_condition):
            - candidates_exist: True if at least 1 candidate exists
            - count: Number of candidates found (limited to 1 for efficiency)
            - filter_condition: The SQL WHERE clause used (for logging)
        """
        candidate_config = self.search_config.get("candidate_generation", {})
        filters_config = candidate_config.get("filters", {})

        # Build the exact same dynamic filter expression used during normal search
        # so preflight checks respect the configured exclusion/diversity logic.
        filter_condition, filter_debug_info = build_lancedb_filter(
            filters_config, context
        )

        # Get vector field from config (default: key_embed)
        vector_field = candidate_config.get("query_embedding_field", "key_embed")

        # Create a dummy query (just check if any rows match the filter)
        # We don't need to actually compute similarity scores for this check
        try:
            # Use a simple count query with the filter
            if filter_condition:
                # LanceDB search requires a vector even for this existence probe, so
                # issue a minimal vector query and let the SQL filter decide whether
                # any candidate row survives.
                import numpy as np

                dummy_embedding = np.zeros(768).tolist()  # e5-base has 768 dimensions

                search_query = self.table.search(
                    dummy_embedding, vector_column_name=vector_field
                )
                search_query = search_query.where(filter_condition)
                search_query = search_query.limit(1)

                results = search_query.to_pandas()

                if results.empty:
                    return False, 0, filter_condition
                else:
                    return True, len(results), filter_condition
            else:
                # No filter - check if table has any data at all
                row_count = self.table.count_rows()
                return row_count > 0, min(row_count, 1), None

        except Exception as e:
            # If the check fails, log the error but don't crash
            print(f"[LanceDBClient] ⚠️  Preflight check failed with error: {e}")
            # Assume candidates exist (fail open) to avoid blocking legitimate episodes
            return True, -1, filter_condition
