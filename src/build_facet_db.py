#!/usr/bin/env python3
"""
build_facet_db.py
Production-grade facet enrichment pipeline for conversation evaluation benchmark.
Scales to 5,000+ facets with checkpointing, validation, and observability.

Provider-agnostic via LLMClientFactory - switch providers in config/llm_config.yaml
"""

import os
import re
import json
import time
import random
import logging
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm import tqdm
import yaml

# Local imports - modular architecture
from src.llm_clients.factory import LLMClientFactory
from src.llm_clients.base import LLMResponse
from src.utils.json_parser import extract_json_from_response
from src.utils.validators import FacetSchema

# =============================================================================
# CONFIGURATION & LOGGING SETUP
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with fallback defaults."""
    defaults = {
        "paths": {
            "input_csv": "data/raw/Facets Assignment.csv",
            "output_csv": "data/processed/Enriched_Facets.csv",
            "logs_dir": "logs",
            "review_sample_file": "data/processed/facets_review_sample.csv"
        },
        "enrichment": {
            "batch_size": 5,
            "review_sample_ratio": 0.05
        },
        "validation": {
            "min_definition_length": 10,
            "facet_id_pattern": r"^FACET-\d{3,5}$"
        },
        "deduplication": {
            "enabled": True,
            "embedding_model": "all-MiniLM-L6-v2",
            "similarity_threshold": 0.92
        }
    }
    
    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            # Deep merge with defaults
            for key in defaults:
                if key not in config:
                    config[key] = defaults[key]
                elif isinstance(defaults[key], dict):
                    for subkey in defaults[key]:
                        if subkey not in config[key]:
                            config[key][subkey] = defaults[key][subkey]
        return config
    return defaults


def load_llm_config(config_path: str = "config/llm_config.yaml") -> dict:
    """Load LLM-specific configuration."""
    defaults = {
        "provider": "groq",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 4096,
        "max_retries": 3,
        "batch_delay_sec": 1.0,
        "enable_logprobs": True
    }
    
    if Path(config_path).exists():
        with open(config_path) as f:
            llm_config = yaml.safe_load(f).get("llm", {})
            # Merge with defaults
            for key in defaults:
                if key not in llm_config:
                    llm_config[key] = defaults[key]
        return llm_config
    return defaults


def setup_logging(logs_dir: str) -> logging.Logger:
    """Configure structured logging to file and console."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(logs_dir) / "facet_enrichment.log", mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def clean_facet_name(name: str) -> str:
    """Robust cleaning of facet names with multiple regex passes."""
    if pd.isna(name):
        return ''
    name = str(name).strip()
    # Remove leading numbers, dots, spaces (e.g., "800. Sufi practice" → "Sufi practice")
    name = re.sub(r'^\d+[\.\s]+', '', name)
    # Remove trailing punctuation and colons
    name = re.sub(r'[:\-.,;]+$', '', name)
    # Normalize internal spacing
    name = re.sub(r'\s+', ' ', name)
    # Remove common suffixes that aren't part of the facet name
    name = re.sub(r'\s*:\s*$', '', name)
    return name.strip()


def clean_raw_data(file_path: str, logger: logging.Logger) -> list[str]:
    """Reads the raw CSV, cleans facet names robustly, and removes duplicates."""
    logger.info(f"Reading raw data from {file_path}...")
    df = pd.read_csv(file_path)
    
    if 'Facets' not in df.columns:
        raise ValueError(f"Column 'Facets' not found in {file_path}")
    
    # Apply robust cleaning function
    df['Facets'] = df['Facets'].apply(clean_facet_name)
    
    # Drop empty strings and duplicates
    initial_count = len(df)
    df = df[df['Facets'].str.len() > 0]
    df = df.drop_duplicates(subset=['Facets']).dropna()
    
    cleaned_list = df['Facets'].tolist()
    logger.info(
        f"✅ Cleaning complete. Found {len(cleaned_list)} unique facets "
        f"(removed {initial_count - len(cleaned_list)} duplicates/empty)."
    )
    return cleaned_list


# =============================================================================
# ENRICHMENT WITH RETRY + VALIDATION
# =============================================================================

def build_enrichment_prompt(facet_batch: list[str], start_id: int, invert_rules: str) -> tuple[str, str]:
    """Build system and user prompts for facet enrichment."""
    
    system_prompt = (
        "You are a data processing API. You strictly output valid JSON only. "
        "Do not include markdown, explanations, or extra text."
    )
    
    user_prompt = f"""
    You are an expert ML Data Engineer building a production-grade conversation evaluation dataset.
    I am providing a list of conversational facets. For each facet, generate a highly detailed structured JSON object.
    
    You must return a valid JSON object with a key "facets" containing an array of objects.
    Each object must strictly have these exact keys:
    
    - "facet_name": (string) The exact name I provided, unchanged.
    - "category": (string) Choose ONE: [Linguistic, Pragmatics, Safety, Emotion].
    - "definition": (string) A clear, 1-sentence definition of what this trait looks like in dialogue.
    - "context_scope": (string) Choose ONE: "single_turn" (evaluated from 1 sentence) or "multi_turn" (requires dialogue history).
    - "invert_scale": (boolean) {invert_rules}
    - "score_1": (string) Concrete behavioral definition for the lowest score (1).
    - "score_2": (string) Concrete behavioral definition for score 2.
    - "score_3": (string) Concrete behavioral definition for the midpoint score (3).
    - "score_4": (string) Concrete behavioral definition for score 4.
    - "score_5": (string) Concrete behavioral definition for the highest score (5).
    - "example_low": (string) 1-sentence realistic dialogue example illustrating score 1 or 2.
    - "example_mid": (string) 1-sentence realistic dialogue example illustrating score 3.
    - "example_high": (string) 1-sentence realistic dialogue example illustrating score 4 or 5.

    Facets to process (generate ONE object per facet):
    {json.dumps(facet_batch, indent=2)}
    """
    
    return system_prompt, user_prompt


def enrich_facets_batch(
    facet_batch: list[str],
    start_id: int,
    llm_client,
    llm_config: dict,
    logger: logging.Logger
) -> list[dict]:
    """Sends a batch of facets to LLM with structured prompt and JSON output."""
    
    invert_scale_rules = """
    invert_scale rules (CRITICAL):
    - TRUE if: higher score = more harmful, risky, undesirable (e.g., Dishonesty, Hostility, Impulsiveness, Toxicity)
    - FALSE if: higher score = more beneficial, skilled, desirable (e.g., Compassion, Clarity, Collaboration, Honesty)
    - When in doubt: FALSE (default to positive framing)
    """
    
    system_prompt, user_prompt = build_enrichment_prompt(
        facet_batch, start_id, invert_scale_rules
    )
    
    try:
        response: LLMResponse = llm_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=llm_config.get("temperature", 0.1),
            max_tokens=llm_config.get("max_tokens", 4096),
            logprobs=llm_config.get("enable_logprobs", False)
        )
        
        if response.error:
            logger.error(f"❌ LLM error: {response.error}")
            return []
        
        # Reuse robust JSON parser
        data = extract_json_from_response(response.content)
        return data.get("facets", [])
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return []


def enrich_facets_with_retry(
    facet_batch: list[str],
    start_id: int,
    llm_client,
    llm_config: dict,
    logger: logging.Logger,
    max_retries: Optional[int] = None
) -> list[dict]:
    """Wrapper with exponential backoff retry logic."""
    if max_retries is None:
        max_retries = llm_config.get("max_retries", 3)
        
    for attempt in range(max_retries):
        result = enrich_facets_batch(
            facet_batch, start_id, llm_client, llm_config, logger
        )
        if result:
            return result
        wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
        logger.warning(
            f"⚠️  Batch {start_id} failed (attempt {attempt+1}/{max_retries}). "
            f"Retrying in {wait_time}s..."
        )
        time.sleep(wait_time)
    
    logger.error(f"❌ Batch {start_id} failed after {max_retries} attempts.")
    return []


def validate_facet(
    item: dict,
    expected_id: str,
    logger: logging.Logger
) -> tuple[bool, str, Optional[dict]]:
    """Validates a single facet against Pydantic schema."""
    try:
        # Assign deterministic ID in Python (don't trust LLM)
        item['facet_id'] = expected_id
        item['schema_version'] = "1.0.0"
        validated = FacetSchema(**item)
        return True, "", validated.dict()
    except ValidationError as e:
        return False, str(e), None


# =============================================================================
# DEDUPLICATION UTILITIES (OPTIONAL)
# =============================================================================

def flag_duplicate_facets(
    facets: list[dict],
    logger: logging.Logger,  # ✅ FIXED: No default → comes BEFORE threshold
    threshold: float = 0.92   # ✅ FIXED: Has default → comes LAST
) -> list[str]:
    """Returns list of facet_ids that are semantic near-duplicates using embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        definitions = [f['definition'] for f in facets]
        embeddings = model.encode(definitions)
        sims = cosine_similarity(embeddings)
        
        duplicates = []
        for i in range(len(facets)):
            for j in range(i+1, len(facets)):
                if sims[i][j] > threshold:
                    duplicates.append(facets[j]['facet_id'])
                    logger.warning(
                        f"⚠️  Potential duplicate: {facets[j]['facet_id']} "
                        f"similar to {facets[i]['facet_id']}"
                    )
        return duplicates
    except ImportError:
        logger.info("⚠️  sentence-transformers not installed; skipping deduplication")
        return []
    except Exception as e:
        logger.warning(f"⚠️  Deduplication check failed: {e}")
        return []


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # Load configurations
    config = load_config()
    llm_config = load_llm_config()
    
    # Setup logging
    logger = setup_logging(config["paths"]["logs_dir"])
    
    # Load environment variables
    load_dotenv()
    
    # === Initialize LLM client via factory (provider-agnostic) ===
    provider = llm_config["provider"]
    api_key = os.environ.get(llm_config.get("api_key_env"))
    
    logger.info(f"🔌 Initializing LLM client: {provider}")
    try:
        llm_client = LLMClientFactory.create(
            provider=provider,
            model=llm_config.get("model"),
            api_key=api_key,
            **llm_config.get(provider, {})  # Provider-specific kwargs
        )
    except ValueError as e:
        logger.error(f"❌ Failed to create LLM client: {e}")
        return 1
    
    if not llm_client.is_available():
        logger.error(
            f"❌ LLM client {provider} is not available. "
            f"Check credentials/config."
        )
        return 1
    logger.info(f"✅ Connected to {provider} ({llm_client.provider_name})")
    # === END LLM client initialization ===
    
    # Paths
    input_csv = config["paths"]["input_csv"]
    output_csv = config["paths"]["output_csv"]
    checkpoint_file = Path(output_csv).with_suffix('.checkpoint.json')
    review_sample_file = Path(config["paths"]["review_sample_file"])
    
    # 1. Clean the data
    raw_facets = clean_raw_data(input_csv, logger)
    
    # 2. Load checkpoint if exists
    processed_ids = set()
    enriched_data = []
    
    # First, try to load from existing output CSV (more reliable than checkpoint)
    if Path(output_csv).exists():
        try:
            df_existing = pd.read_csv(output_csv)
            for _, row in df_existing.iterrows():
                facet_id = row['facet_id']
                processed_ids.add(int(facet_id.split('-')[1]))
                enriched_data.append(row.to_dict())
            logger.info(f"📦 Loaded {len(enriched_data)} facets from existing output CSV.")
        except Exception as e:
            logger.warning(f"⚠️  Could not load existing CSV: {e}. Starting fresh.")
    
    # Fallback to checkpoint file if CSV load failed
    if not enriched_data and checkpoint_file.exists():
        try:
            checkpoint = json.loads(checkpoint_file.read_text())
            processed_ids = set(checkpoint.get("processed_ids", []))
            enriched_data = checkpoint.get("enriched_data", [])
            logger.info(f"📦 Loaded checkpoint: {len(processed_ids)} facets already processed.")
        except Exception as e:
            logger.warning(f"⚠️  Could not load checkpoint: {e}. Starting fresh.")
    
    # 3. Process in batches with progress bar
    batch_size = config["enrichment"]["batch_size"]
    total_batches = (len(raw_facets) + batch_size - 1) // batch_size
    
    logger.info(
        f"🚀 Beginning enrichment via {provider} API "
        f"(batch_size={batch_size}, total_batches={total_batches})..."
    )
    
    with tqdm(total=len(raw_facets), desc="Enriching facets") as pbar:
        for i in range(0, len(raw_facets), batch_size):
            batch_start_id = i + 1  # 1-indexed for FACET-001 style IDs
            
            # Skip if already processed
            if batch_start_id in processed_ids:
                pbar.update(len(raw_facets[i:i + batch_size]))
                continue
                
            batch = raw_facets[i:i + batch_size]
            logger.debug(
                f"🔄 Processing batch {i//batch_size + 1}/{total_batches} "
                f"(IDs {batch_start_id}-{batch_start_id + len(batch) - 1})..."
            )
            
            # Enrich with retry
            batch_results = enrich_facets_with_retry(
                batch, batch_start_id, llm_client, llm_config, logger
            )
            
            if not batch_results:
                logger.warning(f"⚠️  Batch {batch_start_id} returned empty. Skipping.")
                pbar.update(len(batch))
                continue
            
            # Validate each result and assign deterministic IDs
            valid_results = []
            for idx, item in enumerate(batch_results):
                expected_id = f"FACET-{batch_start_id + idx:05d}"  # 5-digit padding for scale
                is_valid, error, validated = validate_facet(item, expected_id, logger)
                if is_valid:
                    valid_results.append(validated)
                else:
                    logger.warning(f"⚠️  Validation failed for {expected_id}: {error}")
            
            if valid_results:
                enriched_data.extend(valid_results)
                processed_ids.update(
                    range(batch_start_id, batch_start_id + len(valid_results))
                )
                
                # Save checkpoint after each successful batch (IDs only to avoid bloat)
                checkpoint_data = {
                    "processed_ids": sorted(list(processed_ids)),
                    "last_batch_start": batch_start_id,
                    "timestamp": time.time()
                }
                checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
                logger.info(
                    f"✅ Batch {batch_start_id} saved. Total enriched: {len(enriched_data)}"
                )
            else:
                logger.warning(f"⚠️  No valid results in batch {batch_start_id}.")
            
            # Rate limiting between batches
            time.sleep(llm_config.get("batch_delay_sec", 1.0))
            pbar.update(len(batch))
    
    # 4. Final save
    if enriched_data:
        df_final = pd.DataFrame(enriched_data)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_final.to_csv(output_csv, index=False)
        logger.info(f"\n🎉 Success! Enriched database saved to {output_csv}")
        logger.info(
            df_final[['facet_id', 'facet_name', 'category', 'context_scope']]
            .head()
            .to_string()
        )
        
        # 5. Quality sampling for manual review
        review_ratio = config["enrichment"]["review_sample_ratio"]
        if len(enriched_data) >= 20 and review_ratio > 0:
            review_sample = random.sample(
                enriched_data, max(1, int(len(enriched_data) * review_ratio))
            )
            pd.DataFrame(review_sample).to_csv(review_sample_file, index=False)
            logger.info(
                f"🔍 Saved {len(review_sample)} facets ({review_ratio*100:.1f}%) "
                f"for manual review: {review_sample_file}"
            )
        
        # 6. Optional: Deduplication check
        if config.get("deduplication", {}).get("enabled", True):
            try:
                threshold = config["deduplication"].get("similarity_threshold", 0.92)
                # ✅ FIXED: Correct argument order: facets, logger, threshold
                duplicates = flag_duplicate_facets(enriched_data, logger, threshold)
                if duplicates:
                    logger.warning(
                        f"⚠️  Found {len(duplicates)} potential duplicate facets. "
                        f"Review flagged IDs."
                    )
                    pd.DataFrame(
                        [f for f in enriched_data if f['facet_id'] in duplicates]
                    ).to_csv(
                        Path(config["paths"]["logs_dir"]) / "potential_duplicates.csv",
                        index=False
                    )
            except Exception as e:
                logger.warning(f"⚠️  Deduplication check failed: {e}")
        
        # Cleanup checkpoint after successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("🧹 Checkpoint file cleaned up.")
    else:
        logger.error("\n❌ Failed to generate any enriched data. Check logs above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())