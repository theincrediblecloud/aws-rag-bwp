# src/rag/core/constants.py
import re

GOOD = re.compile(
    r'\b(decision|build|architecture|approach|solution|design|integrat(?:e|ion)|'
    r'workflow|phase|FRE|FAMRiskEvaluator|evaluate SPI|asynchron|IN_PROGRESS|Oberon)\b',
    re.I
)
GOOD_SUMMARY = re.compile(
    r'\b(decision|decided|design|architecture|approach|solution|integrat(?:e|ion)|'
    r'workflow|phase[s]?|FRE|FAMRiskEvaluator|Oberon|real[- ]?time)\b', re.I
)
BAD_START = re.compile(r'^(glossary|requirements|appendix|background)\b', re.I)
TOO_META  = re.compile(r'\b(for more|refer to|see also|details (?:are|in))\b', re.I)
BAD_META  = re.compile(
    r'\b(we only listed|we (?:will|won\'t)|in this (?:doc|section|quip)|'
    r'for additional (?:factors|information)|details (?:are|in)|'
    r'approaches? details|see (?:also )?|refer to)\b', re.I,
)
BAD_PRONOUN_LEAD = re.compile(r'^(?:we|this (?:doc|section|quip))\b', re.I)

KEY_DECISION = re.compile(r'\b(decision|decided|choose|chose)\b', re.I)
HAS_OBERON   = re.compile(r'\bOberon\b', re.I)
HAS_FRE      = re.compile(r'\bFRE\b|\bFAM\s*Risk\s*Evaluator\b', re.I)
HAS_PHASE    = re.compile(r'\bphase[s]?\b|\bPOC\b|\bmanual moderation\b|\breal[- ]?time\b', re.I)
HAS_OUTPUTS  = re.compile(r'\b(APPROVE|REJECT|IN[_ -]?MANUAL[_ -]?REVIEW|RiskOutcome|data lake|Paragon)\b', re.I)
HAS_INPUTS   = re.compile(r'\b(text|image|image URLs?|catalog|S3)\b', re.I)

BAD_LEAD = re.compile(r'^(additionally|furthermore|moreover|background|glossary|requirements)\b', re.I)

MODE_DEF       = re.compile(r'\b(what is|define|definition|purpose|overview|high[- ]?level)\b', re.I)
MODE_FLOW      = re.compile(r'\b(workflow|how (it|this) works?|process|steps?|pipeline|orchestrat(?:e|ion))\b', re.I)
MODE_PROS      = re.compile(r'\b(pros|cons|trade[- ]?offs?|advantages?|disadvantages?)\b', re.I)
MODE_FLOW_HINT = re.compile(r'\b(workflow|process|steps?|orchestrat(?:e|ion)|evaluate SPI|IN_PROGRESS)\b', re.I)
MODE_PROS_HINT = re.compile(r'\b(pros?|cons?|advantages?|disadvantages?|trade[- ]?offs?)\b', re.I)

APP_REGEX   = re.compile(r'\bAPPENDIX\b', re.I)
TABLEY_LINE = re.compile(r'(?:^\s*[A-Z][A-Z0-9 _/-]{3,}\s+Yes\s+No\s*$)|(\.{3,}\s*\d+\b)', re.I | re.M)
ALLCAPS_RUN = re.compile(r'\b(?:[A-Z]{3,}(?:\s+[A-Z0-9&/-]{3,}){1,})\b')
SENT_SPLIT  = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\(])')

EXCERPT_MAX = 180
EXCERPT_MAX_APPENDIX = 140
PREFER_PAGES_BELOW = 8

BAD_PREFIX = re.compile(
    r'^(?:refer to|see (?:also )?|for more (?:background|details))\b',
    re.I,
)
