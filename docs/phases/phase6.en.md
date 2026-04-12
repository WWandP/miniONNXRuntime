# Phase6

## Focus

- how a GPT prompt enters the model
- how baseline text generation runs
- how the KV-cache path runs
- what `output_text`, `full_token_ids`, and `provider execution summary` mean

## Commands

Baseline:

```bash
./scripts/run_phase.sh phase6
```

KV cache:

```bash
./scripts/run_phase.sh phase6-kv
```

## Key Output

- `last_token_topk`
- `full_token_ids`
- `input_text`
- `input_token_ids`
- `output_text`
- `summary`
- `provider execution summary`

## Read More

- The GPT model assets are already included in the repository
- `phase6` is the baseline text-generation view
- `phase6-kv` is the KV-cache view
- For deeper implementation notes, use the GPT development docs in the repository

## Best For

Use this phase when you want to extend the project from vision models to text models.
