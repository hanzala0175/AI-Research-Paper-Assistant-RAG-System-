"""
Groq LLM client, wired in via LangChain's ChatGroq integration.

Groq's LPU inference gives very low-latency generation, which matters a lot
for an interactive research-assistant UX (multi-document comparison prompts
can involve fairly long contexts, and Groq keeps those fast).
"""
from functools import lru_cache

from langchain_groq import ChatGroq

from app import config


@lru_cache(maxsize=1)
def get_llm(temperature: float = None) -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a free key at "
            "https://console.groq.com/keys and put it in your .env file."
        )
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL,
        temperature=temperature if temperature is not None else config.GROQ_TEMPERATURE,
    )
