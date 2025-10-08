from fastapi import FastAPI
from pydantic import BaseModel

from rag.core.config import AppConfig
from rag.core import retriever
from rag.adapters.embeddings_bedrock import BedrockEmbedder
from rag.adapters.llm_bedrock import BedrockClaude, SYSTEM as LLM_SYSTEM
from rag.adapters.vs_opensearch import OpenSearchVectorStore

cfg = AppConfig()
app = FastAPI(title="RAG Chatbot (skeleton)")

# wire stubs
embedder = BedrockEmbedder()
vector = OpenSearchVectorStore(endpoint=cfg.os_endpoint, index=cfg.os_index, dim=8)
llm = BedrockClaude()
vector.ensure_index()


class ChatReq(BaseModel):
    tenant: str = "bwp"
    audience: str = "internal"
    role: str = "engineer"
    session_id: str = "local"
    user_msg: str
    history: list[dict] = []


@app.get("/health")
def health():
    return {"ok": True, "env": cfg.env, "region": cfg.aws_region}


@app.post("/chat")
def chat(req: ChatReq):
    q_text = retriever.condense_query(req.history, req.user_msg)
    q_vec = embedder.embed([q_text])[0]
    docs = vector.hybrid_search(
        q_text, q_vec, {"tenant": req.tenant, "audience": req.audience, "role": req.role}, k=5
    )
    context = "\n\n".join(f"[{i+1}] {d['title']} – {d['snippet']}" for i, d in enumerate(docs))
    prompt = f"Context:\n{context}\n\nUser: {req.user_msg}"
    answer = "".join(llm.answer(LLM_SYSTEM, prompt, stream=False))
    return {"answer": answer, "citations": docs}
