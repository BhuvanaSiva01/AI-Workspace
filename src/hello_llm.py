"""hello_llm.py — Your first OpenAI API call.

Run it from the repo root:
    python src/hello_llm.py "What is RAG in one sentence?"
"""
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env if a .env file exists.
# On Vocareum the key is already set in the environment, so this is a harmless no-op.
load_dotenv()

# Create the OpenAI client — it reads OPENAI_API_KEY from the environment automatically.
client = OpenAI()


def ask(question: str) -> str:
    """Send one question to the LLM and return the answer text."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user",   "content": question},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Say hello in one sentence."
    print(ask(q))
