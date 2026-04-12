#!/usr/bin/env python3

import argparse
import os
import pathlib
import re
import subprocess
from typing import List

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "run_gpt_text.py requires the `transformers` package. Use a Python env with tokenizer support."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic text-generation wrapper for miniort_run_gpt."
    )
    parser.add_argument(
        "--model-dir",
        default="models/gpt2",
        help="Local Hugging Face model/tokenizer directory.",
    )
    parser.add_argument(
        "--model",
        default="models/gpt2/model.sim.onnx",
        help="ONNX model path consumed by miniort_run_gpt.",
    )
    parser.add_argument(
        "--binary",
        default="./build/miniort_run_gpt",
        help="mini runtime GPT binary.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of greedy-decoded tokens to append in one binary invocation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the mini runtime skips any kernel.",
    )
    return parser.parse_args()


def extract_full_token_ids(output: str) -> List[int]:
    match = re.search(r"full_token_ids:\s*\n\[(.*?)\]", output, re.DOTALL)
    if match is None:
        raise RuntimeError("failed to parse full token ids from miniort_run_gpt output")
    payload = match.group(1).strip()
    if not payload:
        return []
    return [int(part.strip()) for part in payload.split(",")]


def run_generation(
    binary: str,
    model: str,
    model_dir: str,
    token_ids: List[int],
    max_new_tokens: int,
    strict: bool,
) -> List[int]:
    kv_prefill_model = pathlib.Path(model_dir) / "model.kv_prefill.onnx"
    kv_decode_model = pathlib.Path(model_dir) / "model.kv_decode.onnx"
    use_kv_cache = kv_prefill_model.exists() and kv_decode_model.exists()

    cmd = [
        binary,
        "--tokens",
        ",".join(str(token_id) for token_id in token_ids),
        "--generate",
        str(max_new_tokens),
        "--quiet",
    ]
    if use_kv_cache:
        cmd.extend(
            [
                "--kv-cache",
                "--kv-cache-prefill-model",
                kv_prefill_model.as_posix(),
                "--kv-cache-decode-model",
                kv_decode_model.as_posix(),
            ]
        )
    else:
        cmd.insert(1, model)
    if strict:
        cmd.append("--strict")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return extract_full_token_ids(result.stdout)


def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(
        pathlib.Path(args.model_dir).resolve().as_posix(),
        local_files_only=True,
    )

    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("prompt encoded to an empty token sequence")

    original_token_ids = list(token_ids)
    token_ids = run_generation(
        args.binary,
        args.model,
        pathlib.Path(args.model_dir).resolve().as_posix(),
        token_ids,
        args.max_new_tokens,
        args.strict,
    )

    print("input_text:")
    print(args.prompt)
    print("\ninput_token_ids:")
    print(original_token_ids)
    print("\nfull_token_ids:")
    print(token_ids)
    print("\noutput_text:")
    print(tokenizer.decode(token_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
