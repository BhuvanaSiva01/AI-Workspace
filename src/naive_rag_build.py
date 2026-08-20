import numpy as np
import sys
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load OPENAI_API_KEY from .env if a .env file exists.
# On Vocareum the key is already set in the environment, so this is a harmless no-op.
load_dotenv()

# Create the OpenAI client — it reads OPENAI_API_KEY from the environment automatically.
client = OpenAI()
# Set up to Build RAG
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

print("Setup ok. Ready to build RAG.")

#Corpus
documents = [
    # Coffee
    {"id": "coffee_espresso", "text": (
        "Espresso is a concentrated form of coffee made by forcing hot water "
        "under about 9 bars of pressure through finely ground coffee beans. "
        "A single shot is typically 25 to 30 millilitres and takes 25 to 30 "
        "seconds to extract. Espresso forms the base of drinks like the "
        "latte, cappuccino, and americano."
    )},
    {"id": "coffee_beans", "text": (
        "Coffee beans come primarily from two species: Arabica and Robusta. "
        "Arabica accounts for about 60 percent of world production and is "
        "prized for its smoother, more nuanced flavour. Robusta contains "
        "roughly twice as much caffeine and has a stronger, more bitter taste. "
        "Most commercial espresso blends mix the two."
    )},
    {"id": "coffee_brewing", "text": (
        "Pour-over coffee uses a filter cone to drip near-boiling water "
        "through medium-ground coffee. It typically brews for three to four "
        "minutes and produces a clean, light-bodied cup. French press coffee, "
        "in contrast, steeps coarse grounds directly in hot water for four "
        "minutes before pressing, producing a heavier, oil-rich cup."
    )},
    # Tea
    {"id": "tea_green", "text": (
        "Green tea is made from unoxidised leaves of Camellia sinensis. It is "
        "steeped in water at around 70 to 80 degrees Celsius for one to three "
        "minutes. Hotter water or longer steeping produces a bitter, astringent "
        "cup. Green tea is high in an antioxidant called EGCG."
    )},
    {"id": "tea_black", "text": (
        "Black tea comes from fully oxidised Camellia sinensis leaves. It is "
        "brewed with water at or near boiling — 95 to 100 degrees Celsius — "
        "for three to five minutes. Popular varieties include Assam, Darjeeling, "
        "and Ceylon. Black tea typically contains more caffeine than green tea."
    )},
    {"id": "tea_oolong", "text": (
        "Oolong tea is partially oxidised, sitting between green and black tea "
        "in strength and colour. It is brewed at 85 to 95 degrees Celsius for "
        "two to four minutes. Oolong leaves are often rolled and can be re-steeped "
        "several times, with each infusion revealing different flavour notes."
    )},
    # Hot chocolate
    {"id": "chocolate_traditional", "text": (
        "Traditional hot chocolate is made from melted dark chocolate stirred "
        "into hot milk. The ratio is usually 30 to 50 grams of chocolate per "
        "200 millilitres of milk. Whisking prevents the chocolate from settling. "
        "Some recipes add a pinch of chilli or cinnamon for warmth."
    )},
    {"id": "chocolate_powder", "text": (
        "Instant hot chocolate uses cocoa powder mixed with sugar, milk powder, "
        "and stabilisers. Adding hot water dissolves the mix in seconds. It is "
        "cheaper and faster than the traditional method but has a thinner mouthfeel "
        "and less intense chocolate flavour."
    )},
    {"id": "chocolate_history", "text": (
        "Hot chocolate originated with the Maya and Aztec civilisations, who "
        "drank it cold and bitter, spiced with chilli. Europeans encountered "
        "cacao in the 16th century and gradually sweetened the drink and "
        "served it hot. It remained a luxury until industrial cocoa processing "
        "made it affordable in the 19th century."
    )},
    # Milk-based drinks
    {"id": "milk_latte", "text": (
        "A caffè latte is made with one shot of espresso and around 200 "
        "millilitres of steamed milk topped with a thin layer of microfoam. "
        "The ratio is roughly one part espresso to five parts milk. A "
        "cappuccino uses the same espresso base but has equal parts milk and "
        "foam, giving it a lighter, airier texture."
    )},
]

print(f"Loaded {len(documents)} documents.")
for d in documents:
    print(f"  {d['id']:25s}  {len(d['text']):3d} chars")

# Chunking - With sliding Window
def chunk_text(text, size=200, overlap=40):
    """Sliding Window over characters"""
    if len(text)<= size:
        return [text]
    chunks, i=[],0
    while i<len(text):
        end=min(i+size, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i= end - overlap
    return chunks
# Chunk every document, build a flat list with source pointer
all_chunks=[]
for doc in documents:
    for chunk_idx, chunk in enumerate(chunk_text(doc["text"])):
        all_chunks.append({
            "chunk_id": f"{ doc['id']}#{chunk_idx}",
            "source_id": doc["id"],
            "text": chunk,
        })
print(f" Total chunks from all {len(documents)} documents: {len(all_chunks)}\n)")

for c in all_chunks[:8]:
    marker = "..." if len(c["text"])==200 else " "
    print( f" {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
    #print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
print(f" .......... and { len(all_chunks) -8 } more") 

# Chunking strategies compared
def chunk_by_sentence(text):
    """Split on sentence boundaries. Simpler than a proper NLP splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def chunk_by_paragraph(text):
    """One chunk per paragraph."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

sample = documents[0]["text"]  # coffee_espresso
print(f"Sample document: {documents[0]['id']} ({len(sample)} chars)\n")
print(f"Text: {sample}\n")
print("═" * 80)

sw_chunks = chunk_text(sample)
print(f"\n1. SLIDING WINDOW (size=200, overlap=40) → {len(sw_chunks)} chunks")
for i, c in enumerate(sw_chunks):
    print(f"    [{i}] ({len(c):3d} chars) {c!r}")

sent_chunks = chunk_by_sentence(sample)
print(f"\n2. BY SENTENCE → {len(sent_chunks)} chunks")
for i, c in enumerate(sent_chunks):
    print(f"    [{i}] ({len(c):3d} chars) {c!r}")

para_chunks = chunk_by_paragraph(sample)
print(f"\n3. BY PARAGRAPH → {len(para_chunks)} chunks")
for i, c in enumerate(para_chunks):
    print(f"    [{i}] ({len(c):3d} chars) {c!r}")

print("\n" + "═" * 80)
print("Discussion:")
print("  - Sliding window: uniform sizes but cuts words/sentences")
print("  - Sentence: clean units but variable size (some too small)")
print("  - Paragraph: cleanest semantically but often too large per chunk")
print("  - Real systems combine strategies. We use sliding window for now.")

