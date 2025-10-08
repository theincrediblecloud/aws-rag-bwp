from typing import Iterable

SYSTEM = "Use ONLY the provided context. Cite sources like [1], [2]. If missing, say what's missing."

class BedrockClaude:
    def answer(self, system: str, prompt: str, stream: bool = True) -> Iterable[str]:
        # stub just echos; we’ll replace with Bedrock streaming
        yield f"[STUB ANSWER]\n{prompt[:200]}..."
