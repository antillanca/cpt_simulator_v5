import asyncio
import httpx
import time

BASE_URL = "http://localhost:8000"

async def simulate_call(i):
    async with httpx.AsyncClient(timeout=30) as client:
        start = time.time()
        payload = {
            "rule_text": "particle.vy = particle.vy + 0.1"
        }
        try:
            resp = await client.post(f"{BASE_URL}/api/rule/simulate", json=payload)
            elapsed = time.time() - start
            detail = ""
            if resp.status_code != 200:
                try:
                    detail = f" - {resp.json().get('detail')}"
                except:
                    detail = f" - {resp.text}"
            print(f"Request {i}: Status {resp.status_code}{detail}, Time: {elapsed:.2f}s")
            return elapsed
        except Exception as e:
            print(f"Request {i}: Failed with {e}")
            return None

async def delayed_ping(i, delay):
    await asyncio.sleep(delay)
    return await check_server_responsive(i)

async def check_server_responsive(i):
    async with httpx.AsyncClient() as client:
        start = time.time()
        try:
            resp = await client.get(f"{BASE_URL}/api/state/math")
            elapsed = time.time() - start
            print(f"  [PING] Status {resp.status_code}, Time: {elapsed:.2f}s")
            return elapsed
        except Exception as e:
            print(f"  [PING] Failed with {e}")
            return None

async def main():
    print("Starting Concurrency Stress Test...")
    print("Note: Ensure the server is running at localhost:8000")
    
    # 1. Start many docker simulations at once
    sim_tasks = [simulate_call(i) for i in range(10)]
    
    # 2. Delayed pings to see if server responds WHILE simulations are running
    ping_tasks = [delayed_ping(i, 0.2 * i) for i in range(5)]
    
    await asyncio.gather(*(sim_tasks + ping_tasks))
    print("\nStress Test Complete.")

if __name__ == "__main__":
    asyncio.run(main())
