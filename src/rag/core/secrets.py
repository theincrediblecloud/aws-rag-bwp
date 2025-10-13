# src/rag/core/secrets.py
import boto3

_sm = boto3.client("secretsmanager")
_cache = {}

def get_secret(arn: str) -> str:
    if not arn:
        return ""
    if arn in _cache:
        return _cache[arn]
    resp = _sm.get_secret_value(SecretId=arn)
    val = resp.get("SecretString", "")
    _cache[arn] = val
    return val
