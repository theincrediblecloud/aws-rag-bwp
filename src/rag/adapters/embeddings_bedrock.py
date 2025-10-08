from typing import List

# later we'll wire boto3 Bedrock; stub keeps API stable
class BedrockEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # Return unit vectors of fixed small dim as stub
        dim = 8
        return [[1.0 / dim] * dim for _ in texts]
