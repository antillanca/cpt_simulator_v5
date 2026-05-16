import httpx
import asyncio
import socket

import pytest

BASE_URL = "http://localhost:8000"


def _server_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _server_available(), reason="local API server is not running")

def test_math_rule():
    print("Testing Relaxed Validator (Math Rule)...")
    async def _run():
        async with httpx.AsyncClient() as client:
            # A rule with NO 'particle' word. Should be allowed now for math functions.
            payload = {
                "rule_text": "function sum(a,b) return a+b end"
            }
            try:
                resp = await client.post(f"{BASE_URL}/api/rule/test", json=payload)
                print(f"Response: {resp.status_code}")
                if resp.status_code == 200:
                    print("✅ Math rule accepted!")
                else:
                    print(f"❌ Math rule rejected: {resp.text}")
            except Exception as e:
                print(f"Error: {e}")

    asyncio.run(_run())

def test_physics_rule():
    print("\nTesting Relaxed Validator (Physics Rule)...")
    async def _run():
        async with httpx.AsyncClient() as client:
            payload = {
                "rule_text": "particle.vx = 5"
            }
            try:
                resp = await client.post(f"{BASE_URL}/api/rule/test", json=payload)
                print(f"Response: {resp.status_code}")
                if resp.status_code == 200:
                    print("✅ Physics rule accepted!")
                else:
                    print(f"❌ Physics rule rejected: {resp.text}")
            except Exception as e:
                print(f"Error: {e}")

    asyncio.run(_run())

if __name__ == "__main__":
    test_math_rule()
    test_physics_rule()
