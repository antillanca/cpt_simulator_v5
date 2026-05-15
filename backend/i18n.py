"""Translation layer for CPT Simulator v5.

Internal text is ALWAYS English.
Web UI responses are translated to Spanish via Google Translate free API.
CLI/agent responses stay in English.
"""
import httpx
import logging
from backend.config import TRANSLATE_API_URL, TRANSLATE_SOURCE_LANG, TRANSLATE_TARGET_LANG

logger = logging.getLogger(__name__)

# In-memory cache to avoid redundant API calls and reduce latency
TRANSLATION_CACHE = {}


async def translate(text: str, target: str = None, source: str = None) -> str:
    """Translate text using Google Translate free API (client=gtx) asynchronously.
    
    Uses an in-memory cache for speed.
    Returns original text on failure (graceful degradation).
    """
    if not text or not text.strip():
        return text
    
    src = source or TRANSLATE_SOURCE_LANG
    tgt = target or TRANSLATE_TARGET_LANG
    
    if src == tgt:
        return text

    # Check cache first
    cache_key = f"{src}:{tgt}:{text}"
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                TRANSLATE_API_URL,
                params={
                    "client": "gtx",
                    "sl": src,
                    "tl": tgt,
                    "dt": "t",
                    "q": text,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                # Response format: [[[translated_text, original_text, ...], ...], ...]
                translated = "".join(
                    segment[0] for segment in data[0] if segment[0]
                )
                # Store in cache
                TRANSLATION_CACHE[cache_key] = translated
                return translated
    except Exception as e:
        logger.warning(f"[i18n] Translation failed: {e}")
    
    return text  # Fallback: return original


async def translate_dict(d: dict, target: str = None) -> dict:
    """Translate string values in a dict (async). Recurses into nested dicts.
    Non-string values are left untouched.
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = await translate(value, target=target)
        elif isinstance(value, dict):
            result[key] = await translate_dict(value, target=target)
        else:
            result[key] = value
    return result
