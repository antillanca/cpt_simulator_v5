"""TutorEngine - Lua rule generation via cascading LLM providers (async).

Strategy: NVIDIA GLM-5.1 (primary) -> OpenRouter (fallback) -> Ollama local (last resort)
All internal text is English. Translation to Spanish happens at the API boundary (i18n.py).
"""
import json
import re
import asyncio
import httpx

from backend.config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL,
    NVIDIA_MAX_TOKENS, NVIDIA_TEMPERATURE, NVIDIA_TIMEOUT,
    NVIDIA_MAX_RETRIES, NVIDIA_RETRY_DELAY,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODELS,
    OPENROUTER_MAX_TOKENS, OPENROUTER_TEMPERATURE, OPENROUTER_TIMEOUT,
    OLLAMA_FALLBACK_MODEL,
)

SYSTEM_PROMPT = (
    "You are a Lua physics rule generator for CPT Simulator v5. "
    "The simulator has a 'particle' object with x, y, vx, vy properties. "
    "Canvas bounds: x=[0,800], y=[0,600]. Origin top-left. "
    "CRITICAL: Never shadow the 'particle' variable with 'local particle = {}'. "
    "Always READ from particle (e.g. local vx = particle.vx) and WRITE to it (particle.x = result). "
    "Structure your response: first, use <think> tags to reason about the physics logic. "
    "Then, provide ONLY the raw Lua code outside the tags. No markdown code blocks, no backticks, no explanation. "
    "Example: <think>The particle needs to bounce off the floor. I will check y coordinate and invert vy.</think> "
    "if particle.y > 590 then particle.vy = -particle.vy * 0.8 end"
)


def _strip_markdown(text: str) -> str:
    """Clean the model response: remove <think> blocks and markdown fences."""
    # 1. Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # 2. Remove markdown code fences if the model disobeys instructions
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    # 3. Final cleaning of backticks or extra text
    text = text.replace("`", "").strip()
    return text


async def _call_nvidia(messages: list) -> str | None:
    """Call NVIDIA GLM-5.1 via OpenAI-compatible chat API (async)."""
    if not NVIDIA_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "max_tokens": NVIDIA_MAX_TOKENS,
        "temperature": NVIDIA_TEMPERATURE,
    }

    async with httpx.AsyncClient(timeout=NVIDIA_TIMEOUT) as client:
        for attempt in range(NVIDIA_MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    wait = NVIDIA_RETRY_DELAY * (attempt + 1)
                    print(f"[TutorEngine] NVIDIA rate limited, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[TutorEngine] NVIDIA error {resp.status_code}: {resp.text[:200]}")
                    return None
            except httpx.TimeoutException:
                print(f"[TutorEngine] NVIDIA timeout, attempt {attempt+1}")
                await asyncio.sleep(NVIDIA_RETRY_DELAY)
            except Exception as e:
                print(f"[TutorEngine] NVIDIA exception: {e}")
                return None

    return None


async def _call_openrouter(messages: list) -> str | None:
    """Call OpenRouter as fallback (async)."""
    if not OPENROUTER_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cpt-simulator.local",
    }
    # OpenRouter API allows max 3 models in the 'models' array.
    # We chunk the user's long list into groups of 3 and try them sequentially.
    chunks = [OPENROUTER_MODELS[i:i + 3] for i in range(0, len(OPENROUTER_MODELS), 3)]

    for chunk in chunks:
        payload = {
            "models": chunk,
            "messages": messages,
            "max_tokens": OPENROUTER_MAX_TOKENS,
            "temperature": OPENROUTER_TEMPERATURE,
        }

        success_in_chunk = False
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=OPENROUTER_TIMEOUT) as client:
                    resp = await client.post(
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"].get("content")
                        if content:
                            return content
                        else:
                            print(f"[TutorEngine] OpenRouter returned empty content on chunk {chunk}. Trying next models.")
                            break # Try next chunk of models
                    elif resp.status_code == 429:
                        wait_time = [5, 15, 30][attempt]
                        print(f"[TutorEngine] OpenRouter rate limited on chunk {chunk}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"[TutorEngine] OpenRouter error {resp.status_code} on chunk {chunk}: {resp.text[:200]}")
                        break # Try next chunk of models
            except Exception as e:
                print(f"[TutorEngine] OpenRouter exception on chunk {chunk}: {e}")
                await asyncio.sleep(5)
                
        # If we got a result, the function would have returned.
        # If we are here, this chunk failed all attempts. Move to the next chunk.

    return None


async def _call_ollama(prompt: str) -> str | None:
    """Call local Ollama as last resort (async)."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": OLLAMA_FALLBACK_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 512, "temperature": 0.2}
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json().get("response")
            else:
                print(f"[TutorEngine] Ollama error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            import traceback
            print(f"[TutorEngine] Ollama exception ({type(e).__name__}): {e}")
            # traceback.print_exc() # Uncomment if needed
            return None


class TutorEngine:
    """Generates and refines Lua physics rules using cascading LLM providers."""

    async def get_student_prompt(self, objective: str, current_state: dict) -> str:
        """Generate a concise prompt for the StudentEngine to produce Lua code.
        This method does *not* generate the Lua code itself; it only prepares the
        instruction that the StudentEngine will feed to the LLM.
        """
        # Simple prompt construction – we embed the objective and a tiny snapshot
        # of the current state for context. Feel free to enrich later.
        state_json = json.dumps(current_state, ensure_ascii=False)
        prompt = (
            f"Objective: {objective}\n"
            f"Current State: {state_json}\n"
            "Write a clear, concise prompt that a Lua‑generating AI can understand. "
            "Do NOT include any Lua code, just the description of what should be generated."
        )
        return prompt

    async def generate_rule(self, objective: str, current_state: dict, custom_prompt: str = None) -> str | None:
        """Generate a Lua rule for the given objective (async)."""
        if custom_prompt:
            user_msg = custom_prompt
        else:
            user_msg = (
                f"Objective: {objective}\n"
                f"Current State: {json.dumps(current_state)}\n\n"
                f"Write a Lua rule that achieves this objective. "
                f"Output ONLY raw Lua code."
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Try OpenRouter first (User requested skipping NVIDIA)
        result = await _call_openrouter(messages)
        if result:
            return _strip_markdown(result)

        # Last resort: Ollama local
        prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        result = await _call_ollama(prompt)
        if result:
            return _strip_markdown(result)

        print("[TutorEngine] All LLM providers failed.")
        return None

    async def refine_rule(self, failed_rule: str, error: str, objective: str) -> str | None:
        """Refine a rule that failed validation or execution (async)."""
        user_msg = (
            f"Objective: {objective}\n"
            f"Failed Rule: {failed_rule}\n"
            f"Error: {error}\n\n"
            f"Fix the Lua rule to solve the error and meet the objective. "
            f"Output ONLY raw Lua code."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Try OpenRouter first
        result = await _call_openrouter(messages)
        if result:
            return _strip_markdown(result)

        prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        result = await _call_ollama(prompt)
        if result:
            return _strip_markdown(result)

        print("[TutorEngine] All LLM providers failed for refinement.")
        return None


tutor_engine = TutorEngine()
