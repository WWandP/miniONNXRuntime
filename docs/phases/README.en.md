# Phase Guide

These docs are user-facing and focus on three things:

- how to run each phase
- what to look at in the output
- which phase to read next

## Order

1. [phase1](./phase1.en.md): static graph structure
2. [phase2](./phase2.en.md): minimal execution pipeline
3. [phase3](./phase3.en.md): end-to-end inference
4. [phase4](./phase4.en.md): graph optimization and memory
5. [phase5](./phase5.en.md): Execution Provider
6. [phase6](./phase6.en.md): GPT text generation and KV cache

## Unified Entry

All phases use the same script:

```bash
./scripts/run_phase.sh <phase>
```

Common commands:

```bash
./scripts/run_phase.sh phase1
./scripts/run_phase.sh phase3
./scripts/run_phase.sh phase4-opt
./scripts/run_phase.sh phase5
./scripts/run_phase.sh phase6
./scripts/run_phase.sh phase6-kv
```
