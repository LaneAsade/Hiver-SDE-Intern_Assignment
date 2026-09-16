"""
llm_client.py

Thin LLM client wrapping the Gemini API (google-genai SDK), built around
four explicit cost/quota controls:

1. Disk cache, keyed by hash(model, prompt) -- an identical call is NEVER
   re-sent to the API. This is the single biggest lever for "don't burn
   credits": iterating on code/prompts during development replays cached
   responses for free, and a full eval run only pays once.
2. A rate limiter (configurable requests/minute) so a batch run doesn't
   immediately trip 429s -- exact free-tier RPM/RPD varies by model and
   account and is only visible in your AI Studio console
   (https://aistudio.google.com), not in a fixed public number, so this
   defaults conservatively rather than assuming a specific limit.
3. Retry-with-backoff specifically on rate-limit errors.
4. An optional hard `max_calls` circuit breaker: once tripped, it raises
   instead of silently continuing to spend quota.

MockClient implements the same interface for tests and dry runs that need
zero network access and zero quota -- see pipeline.py's --mock flag.

Not implemented here, left as documented next steps (see README): Gemini's
native context caching (~90% off the reused taxonomy/few-shot prefix) and
the Batch API (50% off, best fit for the offline golden-set eval run).
Both are real further savings once the pipeline itself is stable -- adding
them now would be optimizing before the code is even correct.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path("data/llm_cache")


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()


class LLMClient:
    def complete(self, prompt: str, max_tokens: int = 300) -> str:
        raise NotImplementedError


@dataclass
class UsageTracker:
    calls: int = 0
    cache_hits: int = 0
    truncated: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    input_chars: int = 0
    output_chars: int = 0

    def note_live(self, prompt: str, response: str, usage_metadata) -> None:
        self.calls += 1
        self.input_chars += len(prompt)
        self.output_chars += len(response)
        if usage_metadata is not None:
            self.input_tokens += getattr(usage_metadata, "prompt_token_count", 0) or 0
            self.output_tokens += getattr(usage_metadata, "candidates_token_count", 0) or 0
            self.thinking_tokens += getattr(usage_metadata, "thoughts_token_count", 0) or 0

    def note_cached(self, response: str) -> None:
        self.cache_hits += 1

    def summary(self) -> str:
        if self.input_tokens or self.output_tokens:
            thinking = f", {self.thinking_tokens:,} thinking" if self.thinking_tokens else ""
            base = (
                f"{self.calls} live calls, {self.cache_hits} cached "
                f"({self.input_tokens:,} input / {self.output_tokens:,} output tokens{thinking}, real API counts)"
            )
        else:
            est_in, est_out = self.input_chars / 4, self.output_chars / 4
            base = f"{self.calls} live calls, {self.cache_hits} cached (~{est_in:,.0f}/~{est_out:,.0f} tokens, estimated)"
        if self.truncated:
            base += f" -- WARNING: {self.truncated} response(s) hit MAX_TOKENS and may be cut off"
        return base


class GeminiClient(LLMClient):
    """
    Real Gemini API calls. Requires GEMINI_API_KEY in the environment.

    thinking_budget=0 disables "thinking" by default: these are bounded,
    short-output tasks (10-way classification, a ~280-char reply) that
    don't need reasoning, and on Gemini 3.x models max_output_tokens is a
    *combined* budget for thinking + visible output, not separate --
    thinking silently ate an entire draft reply down to a few words
    before this was added. Thinking also bills as output tokens, so
    leaving it on was a real, invisible cost too.
    """

    def __init__(
        self,
        model: str,
        requests_per_minute: int = 10,
        max_calls: int | None = None,
        cache_dir: Path = CACHE_DIR,
        thinking_budget: int = 0,
    ):
        from google import genai  # deferred import: MockClient needs no SDK at all

        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.thinking_budget = thinking_budget
        self.min_interval = 60.0 / requests_per_minute
        self.max_calls = max_calls
        self._last_call_ts = 0.0
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = UsageTracker()

    def _cache_path(self, prompt: str) -> Path:
        return self.cache_dir / f"{_cache_key(self.model, prompt)}.json"

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        cache_path = self._cache_path(prompt)
        if cache_path.exists():
            text = json.loads(cache_path.read_text())["response"]
            self.usage.note_cached(text)
            return text

        if self.max_calls is not None and self.usage.calls >= self.max_calls:
            raise RuntimeError(
                f"Hit the configured max_calls={self.max_calls} live-call budget. "
                f"Raise the limit if this is intentional -- {self.usage.summary()}"
            )

        from google.genai import types

        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                    ),
                )
                self._last_call_ts = time.time()
                text = resp.text or ""

                finish_reason = None
                if getattr(resp, "candidates", None):
                    finish_reason = getattr(resp.candidates[0], "finish_reason", None)
                if finish_reason and "MAX_TOKENS" in str(finish_reason):
                    self.usage.truncated += 1
                    print(
                        f"  WARNING: response truncated (MAX_TOKENS, max_tokens={max_tokens}) -- "
                        f"got {len(text)} chars: {text[:60]!r}..."
                    )

                cache_path.write_text(json.dumps({"prompt": prompt, "response": text}))
                self.usage.note_live(prompt, text, getattr(resp, "usage_metadata", None))
                return text
            except Exception as e:
                rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if rate_limited and attempt < max_retries - 1:
                    backoff = 5 * (2**attempt)  # 5, 10, 20, 40s
                    print(f"  rate limited, backing off {backoff}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff)
                    continue
                raise


class MockClient(LLMClient):
    """
    Deterministic stand-in for tests / pipeline dry runs -- no network
    call, no quota spent. Verifies prompt construction, response parsing,
    and the full classify -> retrieve -> draft -> decide wiring.
    """

    def __init__(self, responder=None):
        self.responder = responder or (lambda prompt: '{"intent": "other", "confidence": 0.5}')
        self.calls: list[str] = []

    def complete(self, prompt: str, max_tokens: int = 300) -> str:
        self.calls.append(prompt)
        return self.responder(prompt)


class OpenAIClient(LLMClient):
    """
    OpenAI-compatible API client (supports Groq, Together, Ollama, vLLM, etc.).
    Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL).
    """
    def __init__(
        self,
        model: str,
        requests_per_minute: int = 10,
        max_calls: int | None = None,
        cache_dir: Path = CACHE_DIR,
    ):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.min_interval = 60.0 / requests_per_minute
        self.max_calls = max_calls
        self._last_call_ts = 0.0
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = UsageTracker()

    def _cache_path(self, prompt: str) -> Path:
        return self.cache_dir / f"{_cache_key(self.model, prompt)}.json"

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        import urllib.request
        import urllib.error

        cache_path = self._cache_path(prompt)
        if cache_path.exists():
            text = json.loads(cache_path.read_text())["response"]
            self.usage.note_cached(text)
            return text

        if self.max_calls is not None and self.usage.calls >= self.max_calls:
            raise RuntimeError(
                f"Hit the configured max_calls={self.max_calls} live-call budget. "
                f"Raise the limit if this is intentional -- {self.usage.summary()}"
            )

        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                req_data = json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }).encode("utf-8")
                
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=req_data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                )

                with urllib.request.urlopen(req) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))

                self._last_call_ts = time.time()
                text = resp_data["choices"][0]["message"]["content"]
                
                usage_metadata = None  # Could map usage if needed
                
                finish_reason = resp_data["choices"][0].get("finish_reason")
                if finish_reason == "length":
                    self.usage.truncated += 1
                    print(
                        f"  WARNING: response truncated (MAX_TOKENS, max_tokens={max_tokens}) -- "
                        f"got {len(text)} chars: {text[:60]!r}..."
                    )

                cache_path.write_text(json.dumps({"prompt": prompt, "response": text}))
                self.usage.note_live(prompt, text, usage_metadata)
                return text
            except urllib.error.HTTPError as e:
                rate_limited = e.code == 429
                if rate_limited and attempt < max_retries - 1:
                    backoff = 5 * (2**attempt)
                    print(f"  rate limited, backing off {backoff}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff)
                    continue
                raise


class OllamaClient(LLMClient):
    """
    Native Ollama API client (uses /api/chat).
    Set OLLAMA_HOST to override the default http://localhost:11434
    """
    def __init__(
        self,
        model: str,
        requests_per_minute: int = 60,  # typically faster/local
        max_calls: int | None = None,
        cache_dir: Path = CACHE_DIR,
    ):
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = model
        self.min_interval = 60.0 / requests_per_minute
        self.max_calls = max_calls
        self._last_call_ts = 0.0
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = UsageTracker()

    def _cache_path(self, prompt: str) -> Path:
        return self.cache_dir / f"{_cache_key(self.model, prompt)}.json"

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        import urllib.request
        import urllib.error

        cache_path = self._cache_path(prompt)
        if cache_path.exists():
            text = json.loads(cache_path.read_text())["response"]
            self.usage.note_cached(text)
            return text

        if self.max_calls is not None and self.usage.calls >= self.max_calls:
            raise RuntimeError(f"Hit max_calls={self.max_calls}")

        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                req_data = json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                }).encode("utf-8")
                
                req = urllib.request.Request(
                    f"{self.host}/api/chat",
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))

                self._last_call_ts = time.time()
                text = resp_data.get("message", {}).get("content", "")
                
                usage_metadata = None
                # Ollama doesn't return stop reason in the same way, but we can check done_reason if present
                if resp_data.get("done_reason") == "length":
                    self.usage.truncated += 1
                    print(f"  WARNING: response truncated. got {len(text)} chars: {text[:60]!r}...")

                cache_path.write_text(json.dumps({"prompt": prompt, "response": text}))
                self.usage.note_live(prompt, text, usage_metadata)
                return text
            except Exception as e:
                print(f"  Ollama request failed: {e}. Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
                if attempt == max_retries - 1:
                    raise
