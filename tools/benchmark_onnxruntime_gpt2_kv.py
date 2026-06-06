#!/usr/bin/env python3
"""Benchmark GPT-2 KV-cache ONNX graphs with Python ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")


DEFAULT_PROMPT_TOKENS = [40, 2883, 6155, 351, 616, 13779, 3290]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def format_stats(values: list[float]) -> str:
    return (
        f"mean={statistics.mean(values):.3f} "
        f"min={min(values):.3f} "
        f"p50={statistics.median(values):.3f} "
        f"p95={percentile(values, 95):.3f} "
        f"max={max(values):.3f}"
    )


def make_session(path: Path, args: argparse.Namespace) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if args.disable_graph_opt
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if args.threads is not None:
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
    providers = args.provider if args.provider else ["CPUExecutionProvider"]
    return ort.InferenceSession(path.as_posix(), sess_options=options, providers=providers)


def select_greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits[0, -1, :]))


def run_once(prefill: ort.InferenceSession, decode: ort.InferenceSession, prompt_tokens: list[int], generate: int):
    token_ids = list(prompt_tokens)
    prefill_input = prefill.get_inputs()[0].name
    decode_input = decode.get_inputs()[0].name

    t0 = time.perf_counter()
    prefill_outputs = prefill.run(None, {prefill_input: np.array([token_ids], dtype=np.int64)})
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    output_names = [output.name for output in prefill.get_outputs()]
    cache = {
        name.replace("present.", "past_key_values."): value
        for name, value in zip(output_names[1:], prefill_outputs[1:])
    }

    decode_ms: list[float] = []
    if generate == 0:
        return prefill_ms, decode_ms, token_ids

    next_token = select_greedy(prefill_outputs[0])
    token_ids.append(next_token)
    step_tokens = [next_token]

    decode_output_names = [output.name for output in decode.get_outputs()]
    for step in range(1, generate + 1):
        feeds = {decode_input: np.array([step_tokens], dtype=np.int64)}
        feeds.update(cache)
        t0 = time.perf_counter()
        decode_outputs = decode.run(None, feeds)
        decode_ms.append((time.perf_counter() - t0) * 1000.0)
        cache = {
            name.replace("present.", "past_key_values."): value
            for name, value in zip(decode_output_names[1:], decode_outputs[1:])
        }
        if step != generate:
            next_token = select_greedy(decode_outputs[0])
            token_ids.append(next_token)
            step_tokens = [next_token]

    return prefill_ms, decode_ms, token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPT-2 KV ONNX Runtime CPU generation.")
    parser.add_argument("--prefill-model", type=Path, default=Path("models/gpt2/model.kv_prefill.onnx"))
    parser.add_argument("--decode-model", type=Path, default=Path("models/gpt2/model.kv_decode.onnx"))
    parser.add_argument("--generate", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--provider", action="append", help="Provider order. Can be passed multiple times.")
    parser.add_argument("--disable-graph-opt", action="store_true")
    parser.add_argument("--tokens", default=",".join(str(token) for token in DEFAULT_PROMPT_TOKENS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_tokens = [int(part) for part in args.tokens.replace(",", " ").split()]
    prefill = make_session(args.prefill_model, args)
    decode = make_session(args.decode_model, args)

    prefill_runs: list[float] = []
    decode_step_runs: list[float] = []
    total_runs: list[float] = []
    final_tokens: list[int] = []

    for run_index in range(args.warmup + args.repeat):
        prefill_ms, decode_ms, token_ids = run_once(prefill, decode, prompt_tokens, args.generate)
        if run_index >= args.warmup:
            prefill_runs.append(prefill_ms)
            decode_step_runs.extend(decode_ms)
            total_runs.append(prefill_ms + sum(decode_ms))
            final_tokens = token_ids

    print("onnxruntime_gpt2_kv_benchmark")
    print(f"ort_version={ort.__version__}")
    print(f"available_providers={ort.get_available_providers()}")
    print(f"active_prefill_providers={prefill.get_providers()}")
    print(f"active_decode_providers={decode.get_providers()}")
    print(f"graph_optimization={'disabled' if args.disable_graph_opt else 'ORT_ENABLE_ALL'}")
    print(f"threads={args.threads if args.threads is not None else 'default'}")
    print(f"warmup={args.warmup}")
    print(f"repeat={args.repeat}")
    print(f"prompt_tokens={len(prompt_tokens)}")
    print(f"generated_tokens={args.generate}")
    print(f"prefill_ms {format_stats(prefill_runs)}")
    print(f"decode_step_ms {format_stats(decode_step_runs)}")
    print(f"total_generation_ms {format_stats(total_runs)}")
    print("full_token_ids:")
    print(final_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
