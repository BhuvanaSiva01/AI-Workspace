from __future__ import annotations
import os
import json
import numpy as np
from openai import OpenAI

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before importing"

_client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"


# ─── Pricing constants ───────────────────────────────────────────────

PRICE_INPUT_PER_1M  = {"gpt-4o-mini": 0.15}
PRICE_OUTPUT_PER_1M = {"gpt-4o-mini": 0.60}
PRICE_EMBED_PER_1M  = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13
}

def cost_usd(result: dict, chat_model: str = CHAT_MODEL) -> float:
    return (
        result["tokens_in"]  * PRICE_INPUT_PER_1M[chat_model]  / 1_000_000 +
        result["tokens_out"] * PRICE_OUTPUT_PER_1M[chat_model] / 1_000_000
    )


# ─── Chunking ────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = end - overlap
    return chunks


def chunk_documents(documents: list[dict], size: int = 200,
                    overlap: int = 40) -> list[dict]:
    all_chunks = []
    for doc in documents:
        for idx, chunk in enumerate(chunk_text(doc["text"], size, overlap)):
            all_chunks.append({
                "chunk_id":  f"{doc['id']}#{idx}",
                "source_id": doc["id"],
                "text":      chunk,
            })
    return all_chunks


# ─── Embedding ───────────────────────────────────────────────────────

def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    resp = _client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def build_index(chunks: list[dict], model: str = EMBED_MODEL) -> list[dict]:
    vectors = embed_batch([c["text"] for c in chunks], model=model)
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec
    return chunks


# ─── Similarity + retrieval ──────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def retrieve(query: str, index: list[dict], k: int = 3,
             embed_model: str = EMBED_MODEL) -> list[dict]:
    q_vec = embed_batch([query], model=embed_model)[0]
    scored = [(cosine(q_vec, c["vector"]), c) for c in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**c, "score": s} for s, c in scored[:k]]


# ─── Prompt + generate ───────────────────────────────────────────────

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say so plainly. "
    "Cite the source id in square brackets after any fact you use."
)

def build_prompt(question: str, retrieved: list[dict],
                 system: str = DEFAULT_SYSTEM) -> tuple[str, str]:
    context = "\n\n".join(
        f"[{hit['chunk_id']}]\n{hit['text']}"
        for hit in retrieved
    )
    user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
    return system, user_msg


def ask_rag(question: str, index: list[dict], k: int = 3,
            system: str = DEFAULT_SYSTEM,
            embed_model: str = EMBED_MODEL,
            chat_model: str = CHAT_MODEL) -> dict:

    retrieved = retrieve(question, index, k=k, embed_model=embed_model)
    system_msg, user_msg = build_prompt(question, retrieved, system)

    resp = _client.chat.completions.create(
        model=chat_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
    )

    return {
        "question":   question,
        "answer":     resp.choices[0].message.content,
        "sources":    [hit["chunk_id"] for hit in retrieved],
        "tokens_in":  resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "retrieved":  retrieved,
    }


# ─── Corpus + golden loading ─────────────────────────────────────────

def load_corpus_from_folder(folder: str) -> list[dict]:
    corpus = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                corpus.append({
                    "id": os.path.splitext(fname)[0],
                    "text": f.read()
                })
    return corpus


def load_golden_set(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# ─── Build corpus + index at import (safe) ───────────────────────────

CORPUS = load_corpus_from_folder("corpus")
chunks = chunk_documents(CORPUS, size=300, overlap=30)
index = build_index(chunks)
