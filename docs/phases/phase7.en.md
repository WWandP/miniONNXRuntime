# Phase7

## Focus

- how to run the default Qwen KV-cache path (prefill/decode dual-graph flow)
- how to read `summary` and `provider execution summary` in the Qwen flow

## Commands

```bash
./scripts/run_phase.sh phase7
```

CUDA build:

```bash
MINIORT_BUILD_CUDA_EP=ON CMAKE_BUILD_TYPE=Release BUILD_DIR=build_cuda_release \
  ./scripts/run_phase.sh phase7
```

With CUDA EP enabled, this uses the current CUDA provider path. The default
`build_local` path is the normal CPU/default build.

## Key Output

- `last_token_topk`
- `full_token_ids`
- `input_text`
- `output_text`
- `summary`
- `provider execution summary`

## Best For

Use this phase when extending the GPT teaching flow to a larger text model (Qwen).
