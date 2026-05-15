import httpx
import asyncio

BASE_URL = "http://localhost:8000"

async def test_math_rule():
    print("Testing Relaxed Validator (Math Rule)...")
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

async def test_physics_rule():
    print("\nTesting Relaxed Validator (Physics Rule)...")
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

if __name__ == "__main__":
    asyncio.run(test_math_rule())
    asyncio.run(test_physics_rule())
