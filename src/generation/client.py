"""Wrapper around the Groq chat completions API."""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (check your .env file)")
        _client = Groq(api_key=api_key)
    return _client


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> str:
    response = get_client().chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
