"""
llm.py
One place that talks to a language model. Three providers:

  LLM_PROVIDER=groq    (default) fast hosted inference; excerpts leave your machine
  LLM_PROVIDER=gemini  Google Gemini API (free tier available); excerpts leave your machine
  LLM_PROVIDER=ollama  fully local (needs `ollama serve` + a pulled model);
                       nothing leaves your machine
"""

import json
import os
import time
import urllib.error
import urllib.request

_groq_client = None

PROVIDER_LABELS = {"groq": "Groq", "gemini": "Google Gemini", "ollama": "Ollama (local)"}


def provider():
    return os.environ.get("LLM_PROVIDER", "groq").lower()


def provider_label():
    return PROVIDER_LABELS.get(provider(), provider())


def is_local():
    return provider() == "ollama"


def _groq(messages, temperature):
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _groq_client = Groq(api_key=key)
    resp = _groq_client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _gemini_payload(messages, temperature):
    """Convert OpenAI-style messages into Gemini's request format."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    payload = {"contents": contents, "generationConfig": {"temperature": temperature}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return payload


def _gemini(messages, temperature, retries=3):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GEMINI_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(_gemini_payload(messages, temperature)).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (503, 429) and attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f"  Gemini busy ({e.code}), retrying in {wait}s... "
                      f"(attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                last_err = RuntimeError(f"Gemini API error {e.code}: {body}")
                continue
            raise RuntimeError(f"Gemini API error {e.code}: {body}") from None
    else:
        raise last_err
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini returned no answer (blocked or empty): {str(data)[:300]}") from None


def _ollama(messages, temperature):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    body = json.dumps({
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"]


def chat(messages, temperature=0.2):
    p = provider()
    if p == "ollama":
        return _ollama(messages, temperature)
    if p == "gemini":
        return _gemini(messages, temperature)
    return _groq(messages, temperature)
