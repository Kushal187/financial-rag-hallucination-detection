"""Wrapper around the Groq chat completions API, with Bedrock as an alternate backend.

`chat_json` forces JSON-object output, and `_retry` backs off on rate limits and transient
5xx errors so a long run doesn't die on the first 429.
"""

import os
import time

import groq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

if _PROVIDER == "bedrock":
    _MODEL = os.getenv("BEDROCK_MODEL", "us.meta.llama3-3-70b-instruct-v1:0")
else:
    _MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client: Groq | None = None

_RateLimitError = getattr(groq, "RateLimitError", ())
_APIStatusError = getattr(groq, "APIStatusError", ())
_APIConnectionError = getattr(groq, "APIConnectionError", ())
_APITimeoutError = getattr(groq, "APITimeoutError", ())


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (check your .env file)")
        _client = Groq(api_key=api_key)
    return _client


def _is_retryable(err: Exception) -> bool:
    """Rate limits, transport errors, and 5xx server errors are worth retrying."""
    if isinstance(err, (_RateLimitError, _APIConnectionError, _APITimeoutError)):
        return True
    status = getattr(err, "status_code", None)
    return isinstance(err, _APIStatusError) and status is not None and status >= 500


def _retry(call_fn, *, max_retries: int = 4, base_delay: float = 2.0, max_delay: float = 60.0):
    """Run ``call_fn`` with exponential backoff on retryable Groq errors."""
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as err:
            if attempt >= max_retries or not _is_retryable(err):
                raise
            time.sleep(min(max_delay, base_delay * (2**attempt)))


def chat(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    response_format: dict | None = None,
) -> str:
    if _PROVIDER == "bedrock":
        from src.generation import bedrock

        return bedrock.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            force_json=bool(response_format),
        )

    def call():
        return get_client().chat.completions.create(
            model=_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    response = _retry(call)
    return response.choices[0].message.content


def chat_json(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Force JSON-object output. Ensures the literal token 'json' is in the prompt
    (Groq rejects ``response_format=json_object`` otherwise)."""
    if not any("json" in str(m.get("content", "")).lower() for m in messages):
        messages = [{"role": "system", "content": "Respond with valid json."}, *messages]
    return chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
