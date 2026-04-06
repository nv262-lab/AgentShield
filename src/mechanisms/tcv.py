"""TCV: Temporal Consistency Verification."""
import numpy as np
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TCVMechanism:
    """Monitor temporal consistency of agent responses across rounds."""

    def __init__(self, window_size: int = 3, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.history = {}  # agent_id -> list of consistency scores

    def update(self, agent_id: int, current_embedding: np.ndarray,
               previous_embedding: np.ndarray = None):
        """Update consistency history for an agent."""
        if agent_id not in self.history:
            self.history[agent_id] = []

        if previous_embedding is not None:
            consistency = float(np.dot(current_embedding, previous_embedding) /
                                 (np.linalg.norm(current_embedding) *
                                  np.linalg.norm(previous_embedding) + 1e-8))
        else:
            consistency = 1.0

        self.history[agent_id].append(consistency)

    def detect_drift(self, agent_id: int) -> Dict:
        """Detect temporal drift for an agent.

        Returns: {agent_id, drift_score, flagged, history}
        """
        if agent_id not in self.history or len(self.history[agent_id]) < 2:
            return {"agent_id": agent_id, "drift_score": 0.0,
                    "flagged": False, "history": []}

        scores = self.history[agent_id][-self.window_size:]
        all_scores = []
        for aid, hist in self.history.items():
            all_scores.extend(hist[-self.window_size:])

        if not all_scores or np.std(all_scores) < 1e-8:
            return {"agent_id": agent_id, "drift_score": 0.0,
                    "flagged": False, "history": scores}

        agent_mean = np.mean(scores)
        pop_mean = np.mean(all_scores)
        pop_std = np.std(all_scores)

        drift = abs(agent_mean - pop_mean) / (pop_std + 1e-8)
        flagged = drift > self.drift_threshold and len(scores) >= 2

        return {
            "agent_id": agent_id,
            "drift_score": float(drift),
            "flagged": flagged,
            "consistency_mean": float(agent_mean),
            "population_mean": float(pop_mean),
            "history": [float(s) for s in scores]
        }

    def reset(self):
        self.history = {}
