# AgentShield: Byzantine Resilience in Multi-Agent LLM Orchestration

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Authors:** Naga Sujitha Vummaneni¹*, Usha Ratnam Jammula²  
¹ Cornell University, nv262@cornell.edu (*corresponding author)  
² Independent Researcher, jammula.usha@gmail.com

**Paper:** IEEE Transactions on Dependable and Secure Computing (TDSC), 2026

## Overview

AgentShield provides Byzantine fault tolerance for multi-agent LLM decision pipelines through three mechanisms:

- **SVEC (Semantic Voting with Equivalence Classes)**: Clusters agent responses by semantic similarity to identify majority consensus
- **TCV (Temporal Consistency Verification)**: Monitors per-agent consistency drift across consensus rounds
- **CAP (Cross-Attestation Probing)**: Verifies suspected agents using factual probes with known answers

## Key Results

| Metric | No Defense | Majority Voting | **AgentShield** |
|--------|-----------|----------------|-----------------|
| CFR (↓) | 67.8% | 52.4% | **2.9%** |
| DA (↑) | 58.3% | 68.1% | **94.7%** |
| Overhead | — | 3 ms | **38 ms** |

## Installation

```bash
git clone https://github.com/nv262-lab/AgentShield.git
cd AgentShield
python3 -m venv .venv
source .venv/bin/activate
pip install "numpy<2" "torch" "transformers==4.40.0" "sentence-transformers==2.7.0"
pip install -r requirements.txt
```

## Quick Start

```python
from src.protocol.agentshield import AgentShield, Agent

shield = AgentShield.from_config("configs/default.yaml")

agents = [
    Agent(0, "Analyst", "gpt-4o", generate_fn=lambda q: "STEMI, recommend PCI"),
    Agent(1, "Reviewer", "claude", generate_fn=lambda q: "Acute STEMI, urgent PCI"),
    Agent(2, "Faulty", "compromised", generate_fn=lambda q: "Patient is fine", is_faulty=True),
    Agent(3, "Verifier", "llama", generate_fn=lambda q: "STEMI confirmed, PCI indicated"),
    Agent(4, "Critic", "mistral", generate_fn=lambda q: "ST elevation MI, proceed with PCI"),
]

result = shield.consensus("Patient: chest pain, troponin 2.4, ST elevation", agents)

print("Decision:", result.decision)           # STEMI consensus
print("Excluded:", result.excluded_agents)     # [2] (faulty agent)
print(f"Overhead: {result.overhead_ms:.1f}ms") # ~38ms
```

## Reproducing Paper Results

```bash
# Run with multiple seeds
python3 scripts/run_main_experiment.py --seed 42 --output results/seed_42.json
python3 scripts/run_main_experiment.py --seed 123 --output results/seed_123.json
python3 scripts/run_main_experiment.py --seed 456 --output results/seed_456.json

# Run unit tests
python3 -m pytest tests/ -v
```

## Project Structure

```
AgentShield/
├── src/
│   ├── mechanisms/        # SVEC, TCV, CAP
│   ├── protocol/          # Full AgentShield BFT consensus
│   ├── evaluation/        # Metrics (CFR, DA, FDR, FAR)
│   └── utils/             # Embeddings
├── configs/default.yaml   # All hyperparameters
├── scripts/               # Reproduction scripts
├── data/                  # Sample datasets (4 domains)
├── tests/                 # Unit tests
└── dataport/              # IEEE DataPort documentation
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.85 | Semantic agreement threshold |
| `tau_s` | 0.15 | TCV drift threshold |
| `alpha_min` | 0.6 | CAP attestation minimum |
| `R_max` | 5 | Maximum consensus rounds |
| `P` | 3 | Number of CAP probes |

## Citation

```bibtex
@article{vummaneni2026agentshield,
  title={Byzantine Resilience in Multi-Agent LLM Orchestration:
         Consensus Protocols for Trustworthy Collective Decision-Making},
  author={Vummaneni, Naga Sujitha and Jammula, Usha Ratnam},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
