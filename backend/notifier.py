import subprocess
import os
import logging
import json
import time
import httpx
import asyncio

logger = logging.getLogger(__name__)

class Notifier:
    """Utility to send notifications using the Hermes Telegram skill."""
    
    NOTIFY_SCRIPT = os.path.expanduser("~/.hermes/scripts/telegram_notify.py")
    GROUP_CHAT_ID = "-5187986760"  # AgenteHermesJA

    def get_token(self):
        # Try env first
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            return token
        # Fallback to .env file in the project
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        return line.split("=", 1)[1].strip()
        # Fallback to hermes .env
        hermes_env = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(hermes_env):
            with open(hermes_env) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        return line.split("=", 1)[1].strip()
        return None

    def send(self, message: str, target: str = "group"):
        """Send a notification to Telegram."""
        if not os.path.exists(self.NOTIFY_SCRIPT):
            logger.warning(f"Notification script not found: {self.NOTIFY_SCRIPT}")
            return False

        try:
            subprocess.run(
                ["python3", self.NOTIFY_SCRIPT, target, message],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def ask_approval(self, message: str, chat_id: str = GROUP_CHAT_ID) -> bool:
        """Sends a message and waits for approval via the Hermes Task Bridge (file-based)."""
        # First, send the notification (we still use sendMessage)
        # We include buttons, but we'll also accept a text reply "aprobar" via the gateway
        token = self.get_token()
        if not token:
            logger.error("TELEGRAM_BOT_TOKEN not found for ask_approval")
            return False

        url_send = f"https://api.telegram.org/bot{token}/sendMessage"
        keyboard = {
            "inline_keyboard": [[
                {"text": "✅ Aprobar", "callback_data": "approve"},
                {"text": "❌ Rechazar", "callback_data": "reject"}
            ]]
        }
        
        payload = {
            "chat_id": chat_id,
            "text": message + "\n\n<i>Responde con 'aprobar' o 'rechazar' para continuar.</i>",
            "parse_mode": "HTML",
            "reply_markup": keyboard
        }

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.post(url_send, json=payload)
                resp.raise_for_status()
                logger.info("Sent approval request to Telegram")
            except Exception as e:
                logger.error(f"Failed to send Telegram approval request: {e}")
                return False

        # Bridge Polling
        bridge_tasks_dir = os.path.expanduser("~/www/.hermes-bridge/tasks")
        os.makedirs(bridge_tasks_dir, exist_ok=True)
        
        # Record start time to ignore old bridge tasks
        start_time = time.time()
        logger.info(f"Starting Bridge approval polling (waiting for files in {bridge_tasks_dir})...")
        
        while True:
            try:
                # Check for new JSON files in bridge tasks
                for fname in sorted(os.listdir(bridge_tasks_dir)):
                    if not fname.endswith(".json"):
                        continue
                    
                    path = os.path.join(bridge_tasks_dir, fname)
                    mtime = os.path.getmtime(path)
                    
                    if mtime >= start_time:
                        with open(path, "r") as f:
                            task = json.load(f)
                            instruction = task.get("instruction", "").lower()
                            
                            if "aprobar" in instruction or "si" == instruction or "yes" == instruction:
                                logger.info(f"Bridge approval received: {instruction}")
                                # Mark task as processed by deleting or renaming? 
                                # Better just return and let orchestrator continue
                                return True
                            elif "rechazar" in instruction or "no" == instruction:
                                logger.info(f"Bridge rejection received: {instruction}")
                                return False
            except Exception as e:
                logger.error(f"Error during Bridge polling: {e}")
            
            await asyncio.sleep(60)

notifier = Notifier()
