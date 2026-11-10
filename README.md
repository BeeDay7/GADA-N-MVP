# GADA-N: Phase-1 Minimum-Viable Benchmark

This repository contains the official code, experiment configurations, and reproducible benchmark for the paper:
**"GADA-N: A Network-Aware Minimum-Viable Benchmark for Autonomous Defense in Military Cyber-Physical Mission Networks"** (submitted to IEEE TNSM).

## Overview
This repository provides the Phase-1 MVP for the GADA-N architecture. It includes:
- The 5-node and 10-node Gymnasium-compatible network topologies.
- The PPO-based autonomous attacker.
- The CTDE-based cooperative defender.
- Meta-behavioral feature extractors (EKF, Decision Entropy, Micro-corrections).

> **Note on Phase-2:** As explicitly stated in the manuscript, the cGAN-based Generative Deception Engine (GDE) and the ns-3/AirSim digital twin are Phase-2 conceptual designs and are **not** included in this Phase-1 release. This repository represents exactly what was evaluated in the paper.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/GADA-N-MVP.git
   cd GADA-N-MVP
