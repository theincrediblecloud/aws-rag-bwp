from typing import Any, Dict, List

class OpenSearchVectorStore:
    def __init__(self, endpoint: str, index: str, dim: int = 8) -> None:
        self.endpoint = endpoint
        self.index = index
        self.dim = dim

    def ensure_index(self) -> None:
        # no-op stub; will create OSS collection + mapping later
        return

    def upsert(self, records: List[Dict[str, Any]]) -> None:
        # no-op stub
        return

    def hybrid_search(
        self, q_text: str, q_vec: List[float], filters: Dict[str, Any], k: int = 10
    ) -> List[Dict[str, Any]]:
        # stub returns fake docs, so UI works before AWS is wired
        tenant = filters.get("tenant", "bwp")
        return [
            {
                "title": f"[stub] {tenant} doc {i+1}",
                "snippet": f"Snippet for query '{q_text}' (stub record {i+1})",
                "source_url": "",
            }
            for i in range(k)
        ]
