"""Unit tests for SVEC mechanism."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mechanisms.svec import SVECMechanism

def test_unanimous_agents():
    svec = SVECMechanism(threshold=0.85)
    embs = np.random.randn(5, 64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    # Make all similar
    base = np.random.randn(64)
    base = base / np.linalg.norm(base)
    embs = np.array([base + np.random.randn(64)*0.01 for _ in range(5)])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    responses = [f"response {i}" for i in range(5)]
    result = svec.cluster(responses, embs)
    assert len(result["majority_class"].members) >= 4

def test_one_outlier():
    svec = SVECMechanism(threshold=0.5)
    base = np.random.randn(64)
    base = base / np.linalg.norm(base)
    embs = [base + np.random.randn(64)*0.01 for _ in range(4)]
    embs.append(np.random.randn(64))  # outlier
    embs = np.array(embs)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    responses = [f"response {i}" for i in range(5)]
    result = svec.cluster(responses, embs)
    assert 4 in result["outliers"] or len(result["outliers"]) >= 1

if __name__ == "__main__":
    test_unanimous_agents()
    test_one_outlier()
    print("All SVEC tests passed!")
