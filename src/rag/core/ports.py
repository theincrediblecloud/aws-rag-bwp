from typing import Any, Dict, List, Protocol, Iterable


class IEmbedder(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class IVectorStore(Protocol):
    def ensure_index(self) -> None: ...
    def upsert(self, records: List[Dict[str, Any]]) -> None: ...
    def hybrid_search(
        self,
        q_text: str,
        q_vec: List[float],
        filters: Dict[str, Any],
        k: int = 10,
    ) -> List[Dict[str, Any]]: ...


class ILLM(Protocol):
    def answer(self, system: str, prompt: str, stream: bool = True) -> Iterable[str]: ...
