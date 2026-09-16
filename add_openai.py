with open(r'src\llm_client.py', 'r') as f:
    content = f.read()

openai_class = '''

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
'''

with open(r'src\llm_client.py', 'w') as f:
    f.write(content + openai_class)
