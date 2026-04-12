# Phase4

## Focus

- graph before/after optimization
- which nodes were folded or removed
- tensor lifetime and memory peak

## Commands

Optimization view:

```bash
./scripts/run_phase.sh phase4-opt
```

Memory view:

```bash
./scripts/run_phase.sh phase4-memory
```

## Key Output

- optimization summary
- node counts before and after optimization
- live tensors / peak bytes
- tensor release timing

## Best For

Use this phase to understand why the graph gets smaller and memory drops.
