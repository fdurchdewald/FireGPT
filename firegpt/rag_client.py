"""
An dieser Stelle bindet ihr eure Retrieval-Augmented-Generation-Logik an.
Das Beispiel ruft einfach das OpenAI-API synchron auf.
"""

from __future__ import annotations
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are FireGPT, ein hilfreicher Assistent für Feuerwehr-Einsätze. "
    "Antworte kurz, präzise und auf Deutsch."
)

def ask_rag(query: str) -> str:
    """Frage an das LLM stellen – hier nur ein simples Beispiel."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
