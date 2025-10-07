from typing import List, Dict


def condense_query(history: List[Dict], user_msg: str) -> str:
    """
    Minimal placeholder: returns user's message as-is.
    Later we'll add a small LLM-based rewrite.
    """
    return user_msg.strip()
