from typing import List, Dict

def condense_query(history: List[Dict], user_msg: str) -> str:
    return user_msg.strip()