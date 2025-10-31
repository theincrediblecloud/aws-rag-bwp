# src/rag/ingest/loaders.py
import os
from pathlib import Path
from typing import Dict, Iterable

def _chunk_text(txt: str, chunk_size: int, overlap: int):
    n = len(txt); i = 0
    while i < n:
        j = min(n, i + chunk_size)
        yield txt[i:j]
        if j == n: break
        i = max(j - overlap, i + 1)

def load_file_to_chunks(path: str) -> Iterable[Dict]:
    """
    Yields dicts with keys:
      text, title, source, page (optional)
    """
    p = Path(path)
    suf = p.suffix.lower()
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    overlap    = int(os.getenv("CHUNK_OVERLAP", "80"))

    if suf == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        for pi, page in enumerate(reader.pages, start=1):
            # Extract text (works for text PDFs; for scans use OCR—see notes)
            txt = page.extract_text() or ""
            if not txt.strip():
                continue
            title = f"{p.stem} (p{pi})"
            for t in _chunk_text(txt, chunk_size, overlap):
                yield {"text": t, "title": title, "source": str(p), "page": pi}

    elif suf == ".docx":
        from docx import Document
        doc = Document(str(p))
        # paragraphs only; tables/images are skipped unless you add custom handling
        txt = "\n".join(para.text for para in doc.paragraphs if para.text)
        if txt.strip():
            for t in _chunk_text(txt, chunk_size, overlap):
                yield {"text": t, "title": p.stem, "source": str(p), "page": None}

    elif suf in {".md", ".txt"}:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if txt.strip():
            for t in _chunk_text(txt, chunk_size, overlap):
                yield {"text": t, "title": p.stem, "source": str(p), "page": None}

    else:
        # Signal unsupported type; caller may log a warning
        raise RuntimeError(f"no loader for {suf}")
