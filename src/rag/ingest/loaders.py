from typing import Iterator, Dict, Any, Iterable
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument
import markdown_it
md = markdown_it.MarkdownIt()

def walk_files(roots: Iterable[str]) -> Iterator[Path]:
    """Yield files under one or many root directories."""
    for root in roots:
        p = Path(root)
        # recurse within each root
        for ext in ("**/*.pdf", "**/*.docx", "**/*.txt", "**/*.md"):
            for f in p.glob(ext):
                if f.is_file():
                    yield f

def load_pdf(path: Path) -> Iterator[Dict[str, Any]]:
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        yield {"title": path.stem, "text": text, "page": i + 1, "source_path": str(path)}


def load_docx(path: Path) -> Iterator[Dict[str, Any]]:
    doc = DocxDocument(str(path))
    paras = [p.text for p in doc.paragraphs]
    text = "\n".join(paras)
    yield {"title": path.stem, "text": text, "page": None, "source_path": str(path)}


def load_txt(path: Path) -> Iterator[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    yield {"title": path.stem, "text": text, "page": None, "source_path": str(path)}


def load_md(path: Path) -> Iterator[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Keep it simple: use raw text; md parser is available if needed for titles
    yield {"title": path.stem, "text": text, "page": None, "source_path": str(path)}


def load_any(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix.lower() == ".pdf":
        yield from load_pdf(path)
    elif path.suffix.lower() == ".docx":
        yield from load_docx(path)
    elif path.suffix.lower() == ".txt":
        yield from load_txt(path)
    elif path.suffix.lower() == ".md":
        yield from load_md(path)