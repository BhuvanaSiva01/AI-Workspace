from __future__ import annotations
import os
import json
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.config import Settings

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


# ─── Chroma Client ───────────────────────────────────────────────────

def get_chroma_client(path="chroma_db"):
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=path
        )
    )


def get_or_create_collection(name="nhanes", path="chroma_db"):
    client = get_chroma_client(path)
    return client.get_or_create_collection(name)


# ─── Chunking ─────────────────────────────────────────────────────────

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


# ─── Save Index to Chroma ────────────────────────────────────────────

def save_index_to_chroma(index: list[dict],
                         collection_name="nhanes",
                         path="chroma_db"):
    collection = get_or_create_collection(collection_name, path)

    ids = [chunk["chunk_id"] for chunk in index]
    embeddings = [chunk["vector"] for chunk in index]
    documents = [chunk["text"] for chunk in index]
    metadatas = [{"source_id": chunk["source_id"]} for chunk in index]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    get_chroma_client(path).persist()
    return collection


# ─── Load Chroma Collection ──────────────────────────────────────────

def load_chroma_collection(collection_name="nhanes", path="chroma_db"):
    client = get_chroma_client(path)
    return client.get_collection(collection_name)


# ─── Retrieval from Chroma ───────────────────────────────────────────

def retrieve_chroma(query: str, collection, k: int = 3,
                    embed_model: str = EMBED_MODEL):
    q_vec = embed_batch([query], model=embed_model)[0]

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=k
    )

    out = []
    for score, doc, meta, id_ in zip(
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["ids"][0]
    ):
        out.append({
            "chunk_id": id_,
            "source_id": meta["source_id"],
            "text": doc,
            "score": score
        })

    return out


# ─── Prompt + Generate ───────────────────────────────────────────────

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


def ask_rag_chroma(question: str, collection, k: int = 3,
                   system: str = DEFAULT_SYSTEM,
                   embed_model: str = EMBED_MODEL,
                   chat_model: str = CHAT_MODEL):

    retrieved = retrieve_chroma(question, collection, k=k)
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
        "cost_usd":   cost_usd({
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens
        }),
        "retrieved":  retrieved,
    }


# ─── Corpus Loading ──────────────────────────────────────────────────

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
