import logging

logger = logging.getLogger(__name__)

"""DPO Pipeline - Onda/Colapso implementation for preference learning.

This module implements the 'Analog Choice' (Onda) and 'Digital Choice' (Colapso)
to generate high-quality Chosen/Rejected pairs for Direct Preference Optimization (DPO).
"""
import json
import asyncio
from backend.ai.student_engine import student
from backend.ai.tutor_engine import tutor_engine
from backend.core.orchestrator import orchestrator

class DPOPipeline:
    def __init__(self, dataset_path="dpo_dataset.jsonl"):
        self.dataset_path = dataset_path

    async def generate_superposition(self, module_name: str, objective: str, current_state: dict, num_variations: int = 3, hints: list = None) -> tuple[list[str], str]:
        """Onda Phase: Generates multiple theoretical variations of Lua rules in parallel."""
        
        # In the DPO pipeline, ALL modules are evaluated against the sandbox particle state.
        # We must always use is_physics=True so the prompt asks for inline particle code,
        # NOT a named function (which the sandbox would never call).
        is_physics = True
        full_prompt = student.build_prompt(module_name, objective, is_physics, hints=hints)

        tasks = []
        for i in range(num_variations):
            variation_prompt = f"{full_prompt}\n\n[Variation Task]: Provide variation #{i+1} using different logic or mathematical approach if possible."
            tasks.append(tutor_engine.generate_rule(objective, current_state, custom_prompt=variation_prompt))
        
        raw_rules = await asyncio.gather(*tasks)
        variations = []
        for rule in raw_rules:
            if rule and rule not in variations:
                variations.append(rule)
        
        return variations, full_prompt

    async def run_collider(self, variations: list[str], target_state: dict, initial_state: dict = None, threshold: float = 0.5) -> list[dict]:
        """Colisionador Phase: Tests all variations in parallel sandboxes."""
        tasks = []
        for rule in variations:
            tasks.append(asyncio.to_thread(orchestrator.simulate_rule, rule, initial_state))
        
        sim_results = await asyncio.gather(*tasks)
        
        results = []
        for rule, sim_result in zip(variations, sim_results):
            score = 0.0
            is_success = False

            if sim_result["status"] == "ok":
                actual_state = sim_result["result"].get("particle", {})
                distance = sum(abs(actual_state.get(k, 0) - target_state.get(k, 0)) for k in target_state)
                score = 1.0 / (1.0 + distance)
                if distance <= threshold:
                    is_success = True

            results.append({
                "rule": rule,
                "score": score,
                "is_success": is_success,
                "error": sim_result.get("message") if sim_result["status"] != "ok" else None
            })

        return results

    def collapse(self, results: list[dict], prompt: str):
        """Colapso Phase: Selects Chosen and Rejected, and saves to DPO dataset."""
        if not results or len(results) < 2:
            return None
            
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        chosen = sorted_results[0]
        rejected = sorted_results[-1]
        
        # If the best one is still a total failure (score 0), skip
        if chosen["score"] <= 0:
            logger.warning(f"[DPO] All variations for '{prompt[:30]}...' failed completely. Skipping.")
            return None
            
        # If chosen and rejected are the same or too close, skip (not enough signal)
        if chosen["score"] <= rejected["score"] * 1.1:
            logger.info(f"[DPO] Signal too weak for '{prompt[:30]}...'. Skipping.")
            return None
            
        dataset_entry = {
            "prompt": prompt,
            "chosen": chosen["rule"],
            "rejected": rejected["rule"],
            "chosen_score": chosen["score"],
            "rejected_score": rejected["score"]
        }
        
        self._save_to_dataset(dataset_entry)
        return dataset_entry
        
    def _save_to_dataset(self, entry: dict):
        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"[DPO] Saved Chosen/Rejected pair to {self.dataset_path}")

dpo_pipeline = DPOPipeline()
