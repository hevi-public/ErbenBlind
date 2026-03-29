"""Model abstraction layer for blinded Erben trials.

Supports multiple backends:
- Claude CLI (`claude --print`) for frontier model calls
- Ollama HTTP API for local model inference

Each call gets a unique nonce appended to defeat prompt-prefix caching.
"""

import json
import re
import subprocess
import urllib.request
import urllib.error
import uuid
from typing import Optional, Tuple


def run_model(prompt: str, model_spec: str, system_prompt: str = "", timeout: int = 600) -> str:
    """Send a prompt to a model and return the response text.

    Each invocation is independent — no conversation memory carries over.
    A unique nonce is appended to prevent prompt-prefix caching.

    Args:
        prompt: The user-message content to send.
        model_spec: Model identifier. Formats:
            - "claude", "sonnet", "opus" → uses claude CLI
            - "ollama:model_name" → uses Ollama HTTP API
        system_prompt: System prompt for the session.
        timeout: Seconds to wait for a response (Ollama only; Claude CLI uses 120s).

    Returns:
        The model's response text (stripped).

    Raises:
        RuntimeError: If the model call fails.
    """
    nonce = uuid.uuid4().hex
    prompt_with_nonce = f"{prompt}\n\n[Analysis ID: {nonce}]"

    if model_spec.startswith("ollama:"):
        model_name = model_spec.split(":", 1)[1]
        return _call_ollama(prompt_with_nonce, model_name, system_prompt, timeout=timeout)
    else:
        return _call_claude(prompt_with_nonce, system_prompt, model_spec)


def is_local_model(model_spec: str) -> bool:
    """Check whether a model spec refers to a local model (no rate limits)."""
    return model_spec.startswith("ollama:")


def strip_thinking_tags(response_text: str) -> Tuple[str, str]:
    """Remove <think>...</think> blocks from DeepSeek-R1 responses.

    The thinking content is returned separately so it can be logged for
    auditing, while the final answer text is parsed for choices.

    Args:
        response_text: Full model response, possibly containing <think> blocks.

    Returns:
        Tuple of (cleaned_response, thinking_content).
        thinking_content is empty string if no <think> block was present.
    """
    think_match = re.search(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
    thinking_content = think_match.group(1).strip() if think_match else ""
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    return cleaned, thinking_content


def _call_claude(prompt: str, system_prompt: str, model: str) -> str:
    """Call the claude CLI as a subprocess with fresh context.

    Args:
        prompt: The user-message content (already has nonce appended).
        system_prompt: The system prompt for the session.
        model: Model name or alias (e.g., "sonnet", "opus").

    Returns:
        The model's response text.

    Raises:
        RuntimeError: If the claude CLI fails.
    """
    cmd = [
        "claude",
        "--print",
        "--model", model,
        "--system-prompt", system_prompt,
        "--no-session-persistence",
    ]

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )

    return result.stdout.strip()


def _call_ollama(prompt: str, model_name: str, system_prompt: str = "", timeout: int = 600) -> str:
    """Call Ollama's chat completion API.

    Uses /api/chat for proper message role support.
    No external dependencies — stdlib urllib only.

    Args:
        prompt: The user-message content (already has nonce appended).
        model_name: Ollama model name (e.g., "qwen2.5:14b").
        system_prompt: Optional system prompt.
        timeout: Seconds to wait for a response.

    Returns:
        The model's response text.

    Raises:
        RuntimeError: If the API call fails or times out.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request_body = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 4096,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=request_body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama API call failed: {e}") from e
    except TimeoutError:
        raise RuntimeError(
            f"Ollama API call timed out after {timeout}s (model: {model_name})"
        )
