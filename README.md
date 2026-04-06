# AgentShield: Byzantine Resilience in Multi-Agent LLM Orchestration

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

AgentShield provides Byzantine fault tolerance for multi-agent LLM decision pipelines through three mechanisms:

- **SVEC**: Semantic Voting with Equivalence Classes
- **TCV**: Temporal Consistency Verification
- **CAP**: Cross-Attestation Probing

## Key Results

| Metric | No Defense | Majority Voting | **AgentShield** |
|--------|-----------|----------------|-----------------|
| CFR (↓) | 67.8% | 52.4% | **2.9%** |
| DA (↑) | 58.3% | 68.1% | **94.7%** |
| Overhead | — | 3 ms | **38 ms** |

## Installation

```bash
git clone https://github.com/[username]/AgentShield.git
cd AgentShield
pip install -r requirements.txt
```

## Quick Start

```python
from src.protocol.agentshield import AgentShield

shield = AgentShield.from_config("configs/default.yaml")
result = shield.consensus(
    task_input="Diagnose: chest pain, elevated troponin, ST elevation",
    agents=agent_list
)
print(result.decision)       # Verified consensus decision
print(result.excluded)       # Excluded faulty agents
print(result.audit_log)      # Full audit trail
```

## Reproducing Results

```bash
bash scripts/reproduce.sh
```

## Citation

```bibtex
@article{agentshield2026,
  title={Byzantine Resilience in Multi-Agent LLM Orchestration: 
         Consensus Protocols for Trustworthy Collective Decision-Making},
  author={[Authors]},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026}
}
```
