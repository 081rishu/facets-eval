#!/usr/bin/env python3
"""
facet_registry.py
Dynamic facet loader for conversation evaluation benchmark.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Generator, Dict, List, Tuple
import yaml
import pandas as pd

logger = logging.getLogger(__name__)


class FacetRegistry:
    """Registry for loading, indexing, and batching facet data."""

    def __init__(self, config_path: Optional[str] = None):
        config_path = config_path or os.environ.get("FACET_CONFIG", "configs/model.yaml")

        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")

        required_keys = ["paths", "inference"]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config key: '{key}' in {config_path}")

        self.facets: List[Dict] = self._load_facets()

        # 🔥 NEW: fast lookup index (O(1))
        self._facet_index: Dict[str, Dict] = {
            f["facet_id"]: f for f in self.facets
        }

        # cache for filtered/grouped results
        self._facet_cache: Dict[str, List[Dict]] = {}

    # -------------------------------------------------------------------------
    # LOADING
    # -------------------------------------------------------------------------

    def _load_facets(self) -> List[Dict]:
        csv_path = self.config['paths']['enriched_facets']
        json_checkpoint_path = str(Path(csv_path).with_suffix('.checkpoint.json'))

        if Path(csv_path).exists():
            logger.info(f"Loading facets from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
        elif Path(json_checkpoint_path).exists():
            logger.warning(f"CSV not found. Loading from checkpoint: {json_checkpoint_path}")
            with open(json_checkpoint_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data.get('enriched_data', []))
        else:
            raise FileNotFoundError(
                f"Facet data not found at {csv_path} or {json_checkpoint_path}"
            )

        required_cols = ["facet_id", "facet_name", "category", "definition"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Loaded {len(df)} total facets")

        return df.to_dict(orient="records")

    # -------------------------------------------------------------------------
    # GROUPED BATCHING (🚀 KEY FOR SCALABILITY)
    # -------------------------------------------------------------------------

    def get_grouped_batches(
        self,
        max_batch_size: Optional[int] = None
    ) -> Generator[Tuple[str, List[Dict]], None, None]:
        """
        Yield batches grouped by category.

        Returns:
            (category, facet_batch)
        """
        if max_batch_size is None:
            max_batch_size = self.config['inference'].get("facet_batch_size", 50)

        # group facets by category
        grouped: Dict[str, List[Dict]] = {}
        for f in self.facets:
            grouped.setdefault(f["category"], []).append(f)

        for category, facets in grouped.items():
            for i in range(0, len(facets), max_batch_size):
                yield category, facets[i:i + max_batch_size]

    # -------------------------------------------------------------------------
    # LEGACY BATCHING (kept for compatibility)
    # -------------------------------------------------------------------------

    def get_batches(
        self,
        batch_size: Optional[int] = None,
        category: Optional[str] = None
    ) -> Generator[List[Dict], None, None]:
        if batch_size is None:
            batch_size = self.config['inference']['facet_batch_size']

        if category:
            if category not in self._facet_cache:
                self._facet_cache[category] = [
                    f for f in self.facets if f.get('category') == category
                ]
            facets = self._facet_cache[category]
        else:
            facets = self.facets

        if not facets:
            logger.warning(f"No facets to yield (category={category})")
            return

        for i in range(0, len(facets), batch_size):
            yield facets[i:i + batch_size]

    # -------------------------------------------------------------------------
    # FAST LOOKUPS
    # -------------------------------------------------------------------------

    def get_facet_by_id(self, facet_id: str) -> Optional[Dict]:
        return self._facet_index.get(facet_id)

    def get_facets_by_category(self, category: str) -> List[Dict]:
        if category not in self._facet_cache:
            self._facet_cache[category] = [
                f for f in self.facets if f.get('category') == category
            ]
        return self._facet_cache[category]

    def __len__(self) -> int:
        return len(self.facets)