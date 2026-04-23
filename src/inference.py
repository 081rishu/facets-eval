#!/usr/bin/env python3
"""
inference.py
Optimized inference engine with grouped batching and async-safe execution.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
import yaml
from dotenv import load_dotenv

from src.llm_clients.factory import LLMClientFactory
from src.llm_clients.base import LLMResponse
from src.utils.json_parser import extract_json_from_response
from src.facet_registry import FacetRegistry
from src.prompt_builder import build_evaluation_prompt
from src.parser import FacetEvaluation, TurnEvaluationResult

logger = logging.getLogger(__name__)
load_dotenv()


class InferenceEngine:
    def __init__(self, config_path: str = "config/llm_config.yaml"):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load full config (NOT just inference)
        with open(config_path) as f:
            full_config = yaml.safe_load(f) or {}

        # Separate configs safely
        self.config = full_config.get("inference", {})
        llm_config = full_config.get("llm", {})

        # -------- API KEY HANDLING (FIXED) --------
        api_key_env = llm_config.get("api_key_env") or "GROQ_API_KEY"
        api_key = os.environ.get(api_key_env)

        if not api_key:
            raise ValueError(
                f"Missing API key. Expected env var: {api_key_env}. "
                f"Check your .env file or environment variables."
            )

        # -------- LLM CLIENT INIT --------
        self.llm_client = LLMClientFactory.create(
            provider=llm_config.get("provider", "groq"),
            model=llm_config.get("model", "llama-3.1-8b-instant"),
            api_key=api_key,
            **llm_config.get(llm_config.get("provider", "groq"), {})
        )

        # -------- REGISTRY --------
        self.registry = FacetRegistry()

        # -------- ASYNC CONFIG --------
        self.semaphore = asyncio.Semaphore(
            self.config.get("max_concurrent_requests", 2)
        )

        self.timeout_sec = self.config.get("request_timeout_sec", 30)

    # ------------------------------------------------------------------
    # CONFIDENCE (simplified + stable)
    # ------------------------------------------------------------------

    def _fallback_confidence(self, response: LLMResponse, n: int) -> List[float]:
        length = len(response.content or "")
        base = min(0.9, 0.5 + length / 4000)
        return [base] * n

    # ------------------------------------------------------------------
    # CORE BATCH EVALUATION
    # ------------------------------------------------------------------

    async def evaluate_batch(
        self,
        turn_text: str,
        facet_batch: List[Dict]
    ) -> List[FacetEvaluation]:

        async with self.semaphore:
            loop = asyncio.get_event_loop()

            for attempt in range(self.config.get("max_retries", 3)):
                try:
                    prompt = build_evaluation_prompt(
                        turn_text,
                        facet_batch,
                        model_context_window=self.config.get("model_context_window", 32768)
                    )

                    # 🔥 FIX: run sync LLM in thread
                    response: LLMResponse = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self.llm_client.generate_json(
                                system_prompt="Return strict JSON only.",
                                user_prompt=prompt,
                                temperature=self.config.get("temperature", 0.1),
                                max_tokens=self.config.get("max_response_tokens", 2048),
                            )
                        ),
                        timeout=self.timeout_sec
                    )

                    if response.error:
                        raise RuntimeError(response.error)

                    data = extract_json_from_response(response.content)
                    results = data.get("results", [])

                    if not results:
                        raise ValueError("Empty results from model")

                    # 🔥 FIX: map by facet_id (NOT index)
                    facet_map = {f["facet_id"]: f for f in facet_batch}

                    confidences = self._fallback_confidence(response, len(results))

                    parsed: List[FacetEvaluation] = []

                    for i, r in enumerate(results):
                        fid = r.get("facet_id")

                        if fid not in facet_map:
                            logger.warning(f"Skipping unknown facet_id: {fid}")
                            continue

                        facet = facet_map[fid]

                        r["invert_scale"] = facet.get("invert_scale", False)
                        r["confidence"] = r.get("confidence", confidences[i])

                        try:
                            parsed.append(FacetEvaluation(**r))
                        except Exception as e:
                            logger.warning(f"Validation failed for {fid}: {e}")

                    return parsed

                except asyncio.TimeoutError:
                    logger.warning(f"⏱ Timeout attempt {attempt+1}")
                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt+1} failed: {e}")

                    if "429" in str(e).lower():
                        wait = (2 ** attempt) + 1
                        await asyncio.sleep(wait)
                        continue
                    else:
                        break

            logger.error("❌ Max retries exceeded for batch")
            return []

    # ------------------------------------------------------------------
    # FULL TURN EVALUATION (OPTIMIZED)
    # ------------------------------------------------------------------

    async def evaluate_turn(self, turn_text: str) -> TurnEvaluationResult:

        # 🔥 NEW: grouped batching (massive efficiency gain)
        grouped_batches = list(self.registry.get_grouped_batches())

        if not grouped_batches:
            return TurnEvaluationResult(
                statement=turn_text,
                total_facets_evaluated=0,
                evaluations=[]
            )

        logger.info(f"→ Evaluating {len(grouped_batches)} grouped batches")

        tasks = [
            self.evaluate_batch(turn_text, batch)
            for _, batch in grouped_batches
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        evaluations: List[FacetEvaluation] = []

        for res in results:
            if isinstance(res, list):
                evaluations.extend(res)
            else:
                logger.error(f"Batch failed: {res}")

        # 🔥 apply invert_scale safely once
        for ev in evaluations:
            if ev.invert_scale:
                ev.score = 6 - ev.score

        return TurnEvaluationResult(
            statement=turn_text,
            total_facets_evaluated=len(evaluations),
            evaluations=evaluations,
            model_used=self.llm_client.provider_name
        )