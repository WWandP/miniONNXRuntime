#!/usr/bin/env python3

import argparse
import html
import json
import os
import re
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SESSION_COOKIE = "miniort_chat_sid"
SYSTEM_PROMPT = "你是一个聊天助手。"


def parse_args() -> argparse.Namespace:
  root_dir = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description="Simple web chat demo backed by miniort_run_qwen.")
  parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
  parser.add_argument("--port", type=int, default=8080, help="HTTP bind port.")
  parser.add_argument(
      "--miniort-bin",
      default=str(root_dir / "build_local" / "miniort_run_qwen"),
      help="Path to miniort_run_qwen binary.",
  )
  parser.add_argument(
      "--model-dir",
      default=str(root_dir / "models" / "qwen2_5_0_5b_instruct"),
      help="Model directory for tokenizer assets.",
  )
  parser.add_argument(
      "--baseline-model",
      default="",
      help="Optional baseline model path. If empty, auto-resolve from model dir.",
  )
  parser.add_argument(
      "--kv-prefill-model",
      default="",
      help="Optional kv prefill model path. If empty, auto-resolve from model dir.",
  )
  parser.add_argument(
      "--kv-decode-model",
      default="",
      help="Optional kv decode model path. If empty, auto-resolve from model dir.",
  )
  parser.add_argument("--generate", type=int, default=48, help="Max new tokens per assistant turn.")
  parser.add_argument("--strict", action="store_true", help="Pass --strict to miniort_run_qwen.")
  parser.add_argument("--timeout-sec", type=int, default=300, help="Timeout for one model call.")
  return parser.parse_args()


def _find_first_existing(candidates: List[Path]) -> Optional[Path]:
  for path in candidates:
    if path.exists():
      return path
  return None


def resolve_model_paths(args: argparse.Namespace) -> Dict[str, str]:
  model_dir = Path(args.model_dir).resolve()
  baseline = Path(args.baseline_model).resolve() if args.baseline_model else _find_first_existing([
      model_dir / "model.baseline.int8.onnx",
      model_dir / "model.baseline.onnx",
  ])
  kv_prefill = Path(args.kv_prefill_model).resolve() if args.kv_prefill_model else _find_first_existing([
      model_dir / "model.kv_prefill.int8.onnx",
      model_dir / "model.kv_prefill.onnx",
  ])
  kv_decode = Path(args.kv_decode_model).resolve() if args.kv_decode_model else _find_first_existing([
      model_dir / "model.kv_decode.int8.onnx",
      model_dir / "model.kv_decode.onnx",
  ])

  if kv_prefill and kv_decode:
    return {
        "mode": "kv",
        "model_dir": str(model_dir),
        "kv_prefill": str(kv_prefill),
        "kv_decode": str(kv_decode),
        "baseline": str(baseline) if baseline else "",
    }
  if baseline:
    return {
        "mode": "baseline",
        "model_dir": str(model_dir),
        "baseline": str(baseline),
        "kv_prefill": "",
        "kv_decode": "",
    }
  raise RuntimeError(
      f"cannot resolve model path under {model_dir}; provide --baseline-model or kv model paths explicitly")


def build_prompt(history: List[Dict[str, str]], user_text: str) -> str:
  lines = [f"System: {SYSTEM_PROMPT}"]
  for turn in history:
    role = turn["role"]
    content = turn["content"]
    if role == "user":
      lines.append(f"User: {content}")
    elif role == "assistant":
      lines.append(f"Assistant: {content}")
  lines.append(f"User: {user_text}")
  lines.append("Assistant:")
  return "\n".join(lines)


def extract_output_text(raw: str) -> str:
  marker = "output_text:\n"
  idx = raw.rfind(marker)
  if idx >= 0:
    return raw[idx + len(marker):].strip()
  return raw.strip()


def extract_assistant_reply(output_text: str, prompt: str) -> str:
  if output_text.startswith(prompt):
    reply = output_text[len(prompt):]
    return reply.strip()
  if "Assistant:" in output_text:
    reply = output_text.split("Assistant:")[-1]
    return reply.strip()
  return output_text.strip()


class ChatRuntime:
  def __init__(self, args: argparse.Namespace, model_paths: Dict[str, str]):
    self.args = args
    self.model_paths = model_paths
    self.sessions: Dict[str, List[Dict[str, str]]] = {}
    self.lock = threading.Lock()

  def get_history(self, sid: str) -> List[Dict[str, str]]:
    with self.lock:
      return list(self.sessions.get(sid, []))

  def append_turn(self, sid: str, role: str, content: str) -> None:
    with self.lock:
      history = self.sessions.setdefault(sid, [])
      history.append({"role": role, "content": content})

  def reset(self, sid: str) -> None:
    with self.lock:
      self.sessions[sid] = []

  def run_model(self, prompt: str) -> Tuple[str, str]:
    cmd = [self.args.miniort_bin]
    if self.model_paths["mode"] == "kv":
      cmd.extend([
          "--kv-cache",
          "--kv-cache-prefill-model", self.model_paths["kv_prefill"],
          "--kv-cache-decode-model", self.model_paths["kv_decode"],
      ])
    else:
      cmd.append(self.model_paths["baseline"])
    cmd.extend([
        "--model-dir", self.model_paths["model_dir"],
        "--prompt", prompt,
        "--generate", str(self.args.generate),
        "--top-k", "5",
        "--quiet",
    ])
    if self.args.strict:
      cmd.append("--strict")

    env = os.environ.copy()
    started = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=self.args.timeout_sec,
        env=env,
        check=False,
    )
    latency = time.time() - started
    if proc.returncode != 0:
      stderr = proc.stderr.strip()
      stdout = proc.stdout.strip()
      detail = stderr if stderr else stdout
      raise RuntimeError(f"miniort_run_qwen failed (code={proc.returncode}): {detail}")

    output_text = extract_output_text(proc.stdout)
    reply = extract_assistant_reply(output_text, prompt)
    if not reply:
      reply = "(empty response)"
    return reply, f"{latency:.2f}s"


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>miniORT Chat Demo</title>
  <style>
    :root {
      --bg: #f4f0e8;
      --panel: #fffaf2;
      --ink: #1b1b1b;
      --sub: #6a6358;
      --accent: #0f766e;
      --user: #fff1cc;
      --assistant: #e9f7ef;
      --border: #d7cdbf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 10% 0%, #fff8eb, var(--bg));
      color: var(--ink);
      font: 16px/1.5 "Avenir Next", "Segoe UI", sans-serif;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .app {
      width: min(900px, 100%);
      height: min(88vh, 900px);
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 18px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.08);
    }
    .head {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .title { font-weight: 700; }
    .meta { color: var(--sub); font-size: 13px; }
    .btn {
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 10px;
      padding: 7px 12px;
      cursor: pointer;
      font-weight: 600;
    }
    .log {
      padding: 18px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      background:
        linear-gradient(135deg, rgba(15,118,110,0.04), transparent 40%),
        linear-gradient(315deg, rgba(251,191,36,0.05), transparent 40%);
    }
    .msg {
      max-width: 82%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      white-space: pre-wrap;
      word-break: break-word;
    }
    .user { align-self: flex-end; background: var(--user); }
    .assistant { align-self: flex-start; background: var(--assistant); }
    .sys { align-self: center; background: #fff; color: var(--sub); font-size: 13px; }
    .composer {
      border-top: 1px solid var(--border);
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }
    textarea {
      width: 100%;
      min-height: 58px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      font: inherit;
      background: #fff;
    }
    .send {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 0 16px;
      font-weight: 700;
      cursor: pointer;
    }
    .send:disabled { opacity: 0.6; cursor: default; }
    @media (max-width: 640px) {
      .app { height: 92vh; border-radius: 12px; }
      .msg { max-width: 92%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="head">
      <div>
        <div class="title">miniORT Chat Demo</div>
        <div class="meta" id="meta">system: 你是一个聊天助手。</div>
      </div>
      <button class="btn" id="resetBtn">重置会话</button>
    </div>
    <div id="log" class="log">
      <div class="msg sys">开始聊天吧。后端会把历史拼接进 prompt。</div>
    </div>
    <form id="form" class="composer">
      <textarea id="input" placeholder="输入你的问题..." required></textarea>
      <button id="sendBtn" class="send" type="submit">发送</button>
    </form>
  </div>
  <script>
    const log = document.getElementById("log");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("sendBtn");
    const resetBtn = document.getElementById("resetBtn");

    function append(role, text) {
      const div = document.createElement("div");
      div.className = "msg " + role;
      div.textContent = text;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    async function send(text) {
      sendBtn.disabled = true;
      append("user", text);
      append("assistant", "思考中...");
      const placeholder = log.lastChild;
      try {
        const resp = await fetch("/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({message: text})
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.error || "request failed");
        }
        placeholder.textContent = data.reply + "\\n\\n[" + data.latency + "]";
      } catch (err) {
        placeholder.textContent = "请求失败: " + err.message;
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      send(text);
    });

    resetBtn.addEventListener("click", async () => {
      await fetch("/reset", {method: "POST"});
      log.innerHTML = "";
      append("sys", "会话已重置。");
    });
  </script>
</body>
</html>
"""


class ChatHandler(BaseHTTPRequestHandler):
  runtime: ChatRuntime = None  # type: ignore

  def _sid(self) -> Tuple[str, bool]:
    raw = self.headers.get("Cookie", "")
    cookie = SimpleCookie()
    cookie.load(raw)
    sid_m = cookie.get(SESSION_COOKIE)
    if sid_m is not None and sid_m.value:
      return sid_m.value, False
    return uuid.uuid4().hex, True

  def _send_json(self, payload: Dict, status: int = 200, set_sid: Optional[str] = None) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    if set_sid:
      self.send_header("Set-Cookie", f"{SESSION_COOKIE}={set_sid}; Path=/; HttpOnly; SameSite=Lax")
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:
    if self.path != "/":
      self.send_error(HTTPStatus.NOT_FOUND, "not found")
      return
    sid, is_new = self._sid()
    body = HTML_PAGE.encode("utf-8")
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    if is_new:
      self.send_header("Set-Cookie", f"{SESSION_COOKIE}={sid}; Path=/; HttpOnly; SameSite=Lax")
    self.end_headers()
    self.wfile.write(body)

  def do_POST(self) -> None:
    if self.path not in ("/chat", "/reset"):
      self.send_error(HTTPStatus.NOT_FOUND, "not found")
      return
    sid, is_new = self._sid()

    if self.path == "/reset":
      self.runtime.reset(sid)
      self._send_json({"ok": True}, set_sid=sid if is_new else None)
      return

    try:
      content_len = int(self.headers.get("Content-Length", "0"))
      raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
      payload = json.loads(raw.decode("utf-8"))
      message = str(payload.get("message", "")).strip()
      if not message:
        self._send_json({"error": "message is empty"}, status=400, set_sid=sid if is_new else None)
        return

      history = self.runtime.get_history(sid)
      prompt = build_prompt(history, message)
      reply, latency = self.runtime.run_model(prompt)
      self.runtime.append_turn(sid, "user", message)
      self.runtime.append_turn(sid, "assistant", reply)
      self._send_json({"reply": reply, "latency": latency}, set_sid=sid if is_new else None)
    except subprocess.TimeoutExpired:
      self._send_json({"error": "model timeout"}, status=504, set_sid=sid if is_new else None)
    except Exception as ex:
      detail = html.escape(str(ex))
      self._send_json({"error": detail}, status=500, set_sid=sid if is_new else None)

  def log_message(self, fmt: str, *args) -> None:
    # Keep logs concise for demo usage.
    print(f"[chat_web_demo] {self.address_string()} - {fmt % args}")


def main() -> None:
  args = parse_args()
  model_paths = resolve_model_paths(args)
  if not Path(args.miniort_bin).exists():
    raise RuntimeError(f"miniort binary not found: {args.miniort_bin}")

  runtime = ChatRuntime(args, model_paths)
  ChatHandler.runtime = runtime
  server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
  mode = model_paths["mode"]
  print(f"chat demo listening on http://{args.host}:{args.port}")
  print(f"miniort: {args.miniort_bin}")
  print(f"mode: {mode}")
  if mode == "kv":
    print(f"kv_prefill: {model_paths['kv_prefill']}")
    print(f"kv_decode:  {model_paths['kv_decode']}")
  else:
    print(f"baseline:   {model_paths['baseline']}")
  server.serve_forever()


if __name__ == "__main__":
  main()
