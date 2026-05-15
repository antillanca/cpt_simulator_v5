import asyncio
import httpx
import time

BASE_URL = "http://localhost:8000"

async def main():
    print("Starting Learning Loop E2E Test...")
    
    async with httpx.AsyncClient(timeout=60) as client:
        # 1. Reset state
        print("Resetting AI state...")
        await client.post(f"{BASE_URL}/api/ai/reset")
        
        # 2. Start learning
        print("Starting learning loop...")
        resp = await client.post(f"{BASE_URL}/api/ai/learn/start")
        print(f"Start response: {resp.json()}")
        
        # 3. Monitor status for 30 seconds
        print("Monitoring status for 30s...")
        for i in range(15):
            status_resp = await client.get(f"{BASE_URL}/api/ai/learn/status")
            status = status_resp.json()
            pending = status.get("pending", 0)
            confirmed = status.get("confirmed", 0)
            print(f"T+{i*2}s: Confirmed: {confirmed}, Pending: {pending}, Is Running: {status.get('is_running')}")
            
            if confirmed > 0:
                print("✅ At least one module confirmed! Progress detected.")
            
            await asyncio.sleep(2)

        # 4. Stop learning
        print("Stopping learning loop...")
        await client.post(f"{BASE_URL}/api/ai/learn/stop")
        print("Test Complete.")

if __name__ == "__main__":
    asyncio.run(main())
