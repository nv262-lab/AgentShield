"""CAP: Cross-Attestation Probing."""
import numpy as np
import time
import logging
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)


class CAPMechanism:
    """Verify suspected agents using factual probes with known answers."""

    def __init__(self, num_probes: int = 3, num_verifiers: int = 2,
                 attestation_min: float = 0.6, probe_db: List[Dict] = None):
        self.num_probes = num_probes
        self.num_verifiers = num_verifiers
        self.attestation_min = attestation_min
        self.probe_db = probe_db or self._default_probes()

    def _default_probes(self) -> List[Dict]:
        return [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is H2O?", "answer": "Water"},
            {"question": "How many days in a week?", "answer": "7"},
            {"question": "What planet is closest to the sun?", "answer": "Mercury"},
        ]

    def probe_agent(self, agent_fn: Callable, suspect_id: int,
                     verifier_fns: List[Callable] = None,
                     similarity_fn: Callable = None) -> Dict:
        """Probe a suspected agent with factual questions.

        Args:
            agent_fn: Function that takes a question and returns answer
            suspect_id: ID of the suspected agent
            verifier_fns: Functions for verifier agents
            similarity_fn: Semantic similarity function

        Returns: {agent_id, attestation_score, correct_count, probes, excluded}
        """
        start = time.time()
        probes = np.random.choice(len(self.probe_db), size=min(self.num_probes, len(self.probe_db)), replace=False)

        correct = 0
        total = 0
        probe_results = []

        for p_idx in probes:
            probe = self.probe_db[p_idx]
            try:
                response = agent_fn(probe["question"])
                ground_truth = probe["answer"]

                if similarity_fn:
                    score = similarity_fn(response, ground_truth)
                    is_correct = score >= 0.7
                else:
                    is_correct = ground_truth.lower() in response.lower()

                if is_correct:
                    correct += 1
                total += 1

                probe_results.append({
                    "question": probe["question"],
                    "response": response,
                    "expected": ground_truth,
                    "correct": is_correct
                })
            except Exception as e:
                logger.warning(f"Probe failed for agent {suspect_id}: {e}")
                total += 1

        attestation = correct / max(total, 1)
        excluded = attestation < self.attestation_min

        elapsed = (time.time() - start) * 1000
        logger.debug(f"CAP: agent {suspect_id} score={attestation:.2f}, "
                      f"excluded={excluded} in {elapsed:.1f}ms")

        return {
            "agent_id": suspect_id,
            "attestation_score": attestation,
            "correct_count": correct,
            "total_probes": total,
            "excluded": excluded,
            "probe_results": probe_results
        }
