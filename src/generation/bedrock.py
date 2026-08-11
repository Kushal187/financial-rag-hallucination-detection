"""Bedrock backend: same Llama model as Groq, no daily token limit.

Its Converse API wants system messages separated out, content as blocks, and alternating
roles, which is what to_converse builds.
"""

import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("BEDROCK_MODEL", "us.meta.llama3-3-70b-instruct-v1:0")
REGION = os.getenv("AWS_REGION", "us-east-1")

_client = None


def get_client():
    """Create the Bedrock client once. boto3 retries throttling for us."""
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
        )
    return _client


def to_converse(messages):
    """Turn Groq-style messages into Bedrock's (system, messages) pair."""
    system = []
    turns = []
    for message in messages:
        text = str(message.get("content", ""))
        if message.get("role") == "system":
            system.append({"text": text})
            continue

        role = "assistant" if message.get("role") == "assistant" else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"][0]["text"] += "\n\n" + text
        else:
            turns.append({"role": role, "content": [{"text": text}]})

    if turns and turns[0]["role"] == "assistant":
        turns.insert(0, {"role": "user", "content": [{"text": "Continue."}]})
    return system, turns


def chat(messages, temperature=0.0, max_tokens=512, force_json=False):
    """Send the messages to Bedrock and return the reply text."""
    system, turns = to_converse(messages)
    if force_json:
        system.append({"text": "Reply with a single valid json object and nothing else."})

    kwargs = {
        "modelId": MODEL,
        "messages": turns,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = system

    response = get_client().converse(**kwargs)
    blocks = response["output"]["message"]["content"]
    return "".join(block.get("text", "") for block in blocks)
