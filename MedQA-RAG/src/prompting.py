SYSTEM_INSTRUCTIONS = """You are a helpful medical assistant.
Answer ONLY using the provided context passages. If the answer is not in the passages, say "I don't know based on the provided sources."
Use concise language and include bracketed citations like [CITATION 1], [CITATION 2] matching the passage numbers you used.
"""

def _truncate_by_chars(text: str, max_chars: int = 700) -> str:
    return (text[:max_chars] + "…") if len(text) > max_chars else text

def build_prompt(question: str, passages: list[str], max_passages: int = 6, per_passage_chars: int = 700) -> str:
    numbered = []
    for i, p in enumerate(passages[:max_passages], 1):
        numbered.append(f"[{i}] {_truncate_by_chars(p, per_passage_chars)}")
    ctx = "\n".join(numbered)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context passages:\n{ctx}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
