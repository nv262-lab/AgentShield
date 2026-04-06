"""AgentShield: Full BFT consensus protocol for multi-agent LLMs."""
import yaml
import time
import logging
import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field
from ..mechanisms.svec import SVECMechanism
from ..mechanisms.tcv import TCVMechanism
from ..mechanisms.cap import CAPMechanism

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    agent_id: int
    name: str
    model: str
    generate_fn: Callable = None
    is_faulty: bool = False
    fault_type: str = ""


@dataclass
class ConsensusResult:
    decision: str
    excluded_agents: List[int] = field(default_factory=list)
    rounds_taken: int = 0
    overhead_ms: float = 0.0
    audit_log: Dict = field(default_factory=dict)
    majority_class_size: int = 0
    total_agents: int = 0


class AgentShield:
    """Full AgentShield BFT protocol: SVEC + TCV + CAP."""

    def __init__(self, config: dict):
        self.config = config
        self.svec = SVECMechanism(
            threshold=config.get("svec", {}).get("agreement_threshold", 0.85)
        )
        self.tcv = TCVMechanism(
            window_size=config.get("tcv", {}).get("window_size", 3),
            drift_threshold=config.get("tcv", {}).get("drift_threshold", 0.15)
        )
        self.cap = CAPMechanism(
            num_probes=config.get("cap", {}).get("num_probes", 3),
            attestation_min=config.get("cap", {}).get("attestation_min", 0.6)
        )
        self.max_rounds = config.get("protocol", {}).get("max_rounds", 5)
        self.embedder = None

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _get_embedder(self):
        if self.embedder is None:
            from ..utils.embeddings import EmbeddingModel
            self.embedder = EmbeddingModel()
        return self.embedder

    def consensus(self, task_input: str, agents: List[Agent]) -> ConsensusResult:
        """Execute full AgentShield consensus protocol."""
        start = time.time()
        embedder = self._get_embedder()
        n = len(agents)
        active = set(range(n))
        audit = {"rounds": [], "exclusions": []}

        # Phase 1: Independent evaluation
        responses = {}
        for i, agent in enumerate(agents):
            if agent.generate_fn:
                responses[i] = agent.generate_fn(task_input)
            else:
                responses[i] = f"Agent {agent.name} response to: {task_input[:50]}"

        prev_embeddings = {}

        for round_num in range(1, self.max_rounds + 1):
            round_log = {"round": round_num, "active": list(active)}

            # Get current responses and embeddings
            active_responses = [responses[i] for i in sorted(active)]
            active_ids = sorted(active)
            embeddings = embedder.encode(active_responses)

            # Phase 2a: SVEC clustering
            svec_result = self.svec.cluster(active_responses, embeddings)
            majority = svec_result["majority_class"]
            suspects = set(active_ids[i] for i in svec_result["outliers"])

            # Phase 2b: TCV drift detection
            for idx, aid in enumerate(active_ids):
                prev = prev_embeddings.get(aid)
                self.tcv.update(aid, embeddings[idx], prev)
                prev_embeddings[aid] = embeddings[idx]
                drift = self.tcv.detect_drift(aid)
                if drift["flagged"]:
                    suspects.add(aid)

            round_log["suspects"] = list(suspects)

            # Phase 2c: CAP probing for suspects
            if suspects:
                for sid in list(suspects):
                    agent = agents[sid]
                    cap_result = self.cap.probe_agent(
                        agent_fn=agent.generate_fn or (lambda q: "unknown"),
                        suspect_id=sid
                    )
                    if cap_result["excluded"]:
                        active.discard(sid)
                        audit["exclusions"].append({
                            "agent_id": sid, "round": round_num,
                            "attestation": cap_result["attestation_score"]
                        })

            audit["rounds"].append(round_log)

            # Check termination
            active_in_majority = sum(1 for m in majority.members
                                      if active_ids[m] in active)
            f_max = (n - 1) // 3
            if active_in_majority >= len(active) - f_max:
                break

        # Phase 3: Decision aggregation
        decision = majority.representative if majority else ""

        elapsed = (time.time() - start) * 1000
        return ConsensusResult(
            decision=decision,
            excluded_agents=[e["agent_id"] for e in audit["exclusions"]],
            rounds_taken=round_num,
            overhead_ms=elapsed,
            audit_log=audit,
            majority_class_size=len(majority.members) if majority else 0,
            total_agents=n
        )
