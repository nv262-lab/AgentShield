"""SVEC: Semantic Voting with Equivalence Classes."""
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EquivalenceClass:
    members: List[int] = field(default_factory=list)
    centroid: np.ndarray = None
    representative: str = ""


class SVECMechanism:
    """Cluster agent responses by semantic similarity and identify majority."""

    def __init__(self, similarity_fn=None, threshold: float = 0.85):
        self.similarity_fn = similarity_fn
        self.threshold = threshold

    def cluster(self, responses: List[str], embeddings: np.ndarray) -> Dict:
        """Cluster responses into semantic equivalence classes.

        Returns: {
            "classes": List[EquivalenceClass],
            "majority_class": EquivalenceClass,
            "outliers": List[int],
            "similarity_matrix": np.ndarray
        }
        """
        start = time.time()
        n = len(responses)
        sim_matrix = embeddings @ embeddings.T

        # Greedy clustering (order by avg similarity to all others)
        avg_sims = sim_matrix.mean(axis=1)
        order = np.argsort(-avg_sims)

        classes = []
        assigned = set()

        for idx in order:
            if idx in assigned:
                continue
            new_class = EquivalenceClass(members=[int(idx)],
                                          representative=responses[idx])
            assigned.add(idx)
            for other in order:
                if other in assigned:
                    continue
                if sim_matrix[idx][other] >= self.threshold:
                    new_class.members.append(int(other))
                    assigned.add(other)
            new_class.centroid = embeddings[new_class.members].mean(axis=0)
            classes.append(new_class)

        # Majority = largest class
        majority = max(classes, key=lambda c: len(c.members))
        outliers = [i for i in range(n) if i not in set(majority.members)]

        elapsed = (time.time() - start) * 1000
        logger.debug(f"SVEC: {len(classes)} classes, majority={len(majority.members)}, "
                      f"outliers={len(outliers)} in {elapsed:.1f}ms")

        return {
            "classes": classes,
            "majority_class": majority,
            "outliers": outliers,
            "similarity_matrix": sim_matrix.tolist()
        }
