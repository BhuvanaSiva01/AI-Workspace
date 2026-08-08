"""pydantic_demo.py — Live-demo script for W2 Topic 1 (Pydantic).

Three demos, one per deck slide:

    basic  (slide 8)   BaseModel, type hints, model_dump(), coercion
    bad    (slide 9)   what a ValidationError actually looks like
    ai     (slide 10)  parse an LLM JSON response with ExpectedAnswer

Run them all at once:

    python pydantic_demo.py

Or one at a time during teaching (recommended for the live session):

    python pydantic_demo.py basic
    python pydantic_demo.py bad
    python pydantic_demo.py ai
"""
from __future__ import annotations

import json
import sys
from typing import Optional

from pydantic import BaseModel, ValidationError


def _banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")


# ----------------------------------------------------------------------
# Demo 1 — A basic Pydantic model  (slide 8)
# ----------------------------------------------------------------------
def demo_basic() -> None:
    _banner("Demo 1 — A basic Pydantic model")

    class User(BaseModel):
        name: str
        age: int
        email: Optional[str] = None

    # Construct from kwargs — the shape you'll see on every slide.
    u = User(name="Asha", age=29)
    print("Construct from kwargs:")
    print(f"  u            = {u}")
    print(f"  u.model_dump() -> {u.model_dump()}\n")

    # Pydantic is happy with *coercible* inputs (note: int from the string '30').
    u2 = User(name="Bo", age="30")
    print("Pydantic coerces compatible types — strict on invalid, lenient on coercible:")
    print(f"  User(name='Bo', age='30')   ->  {u2}")
    print(f"  type(u2.age).__name__       ->  {type(u2.age).__name__!r}\n")

    # Optional fields default to None — no missing-key surprises.
    u3 = User(name="Cai", age=42)
    print("Optional fields with defaults just work:")
    print(f"  u3.model_dump() -> {u3.model_dump()}\n")


# ----------------------------------------------------------------------
# Demo 2 — When bad data goes in  (slide 9)
# ----------------------------------------------------------------------
def demo_bad_data() -> None:
    _banner("Demo 2 — When bad data goes in")

    class User(BaseModel):
        name: str
        age: int
        email: Optional[str] = None

    print("Try to construct User with age='not a number':\n")
    try:
        User(name="Asha", age="not a number")
    except ValidationError as exc:
        print(exc)

    print("\nWhat the error tells you, in three lines:")
    print("  - WHICH field broke   ->  age")
    print("  - WHY it broke        ->  int_parsing failed")
    print("  - WHAT you sent       ->  'not a number'")
    print("\nNo silent corruption. No untyped dict drifting through the system.\n")


# ----------------------------------------------------------------------
# Demo 3 — Parse an LLM response  (slide 10)
# ----------------------------------------------------------------------
def demo_ai_use_case() -> None:
    _banner("Demo 3 — Parse an LLM JSON response")

    class ExpectedAnswer(BaseModel):
        summary: str
        confidence: float
        sources: list[str]

    # ----- Case A: well-formed JSON -----
    good = json.dumps({
        "summary": "RAG combines retrieval over a corpus with an LLM generator.",
        "confidence": 0.9,
        "sources": ["doc1", "doc2"],
    })
    print("Case A — well-formed JSON from the model:")
    print(f"  raw    = {good}")
    parsed = ExpectedAnswer.model_validate_json(good)
    print(f"  parsed = {parsed}\n")
    print(f"  parsed.summary    -> {parsed.summary}")
    print(f"  parsed.confidence -> {parsed.confidence}")
    print(f"  parsed.sources    -> {parsed.sources}\n")

    # ----- Case B: wrong type for one field -----
    bad_type = json.dumps({
        "summary": "RAG combines retrieval over a corpus with an LLM generator.",
        "confidence": "high",     # should be a float
        "sources": ["doc1"],
    })
    print("Case B — wrong type for `confidence` ('high' instead of a float):\n")
    try:
        ExpectedAnswer.model_validate_json(bad_type)
    except ValidationError as exc:
        print(exc)
    print()

    # ----- Case C: missing field -----
    missing = json.dumps({
        "summary": "RAG combines retrieval over a corpus with an LLM generator.",
        "confidence": 0.9,
        # `sources` missing entirely
    })
    print("Case C — `sources` field missing entirely:\n")
    try:
        ExpectedAnswer.model_validate_json(missing)
    except ValidationError as exc:
        print(exc)
    print()

    print("Takeaway:")
    print("  Every LLM response that should be structured passes through a model like this.")
    print("  When the model returns something off-shape, you find out at the door — not")
    print("  three function calls later when something else explodes.\n")


# ----------------------------------------------------------------------
DEMOS = {
    "basic": demo_basic,
    "bad":   demo_bad_data,
    "ai":    demo_ai_use_case,
}


def main() -> None:
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if arg == "all":
        for fn in DEMOS.values():
            fn()
    elif arg in DEMOS:
        DEMOS[arg]()
    else:
        print(f"Unknown demo: {arg!r}")
        print(f"Choose one of: {', '.join(DEMOS)}, or omit for all.")
        sys.exit(1)


if __name__ == "__main__":
    main()
