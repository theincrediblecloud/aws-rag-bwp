from typing import List
import re

PARA_SPLIT = re.compile(r"\n\s*\n+")  # blank-line paragraph breaks

def split_into_paras(text: str) -> List[str]:
    # split on blank lines first for coherence
    parts = PARA_SPLIT.split(text or "")
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]
    return cleaned if cleaned else [text.strip()]

def chunk_text(text: str, size: int = 900, overlap: int = 60) -> List[str]:
    """Simple word-window chunking: ~size words with overlap."""
    paras = split_into_paras(text)
    words: List[str] = []
    for p in paras:
        words.extend(p.split())

    chunks: List[str] = []
    if not words:
        return chunks

    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        end = min(len(words), start + size)
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks
