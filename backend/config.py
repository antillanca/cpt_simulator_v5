import os
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

# === Simulation ===
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
DEFAULT_STATE = {"x": 300, "y": 50, "vx": 0, "vy": 0}
SIM_STEP_DELAY = 0.016  # ~60fps
SIM_MULTISTEP_FRAMES = 200  # Frames por evaluación de regla en learning loop

# === Database ===
DATABASE_URL = os.getenv("CPT_DATABASE_URL", "sqlite:///./rules.db")

# === Sandbox Docker ===
SANDBOX_IMAGE = "cpt-sandbox"
SANDBOX_MEMORY = "64m"
SANDBOX_CPUS = 0.2
SANDBOX_PIDS_LIMIT = 32
SANDBOX_TIMEOUT_MS = 3000

# === LLM Strategy (cascading fallback) ===
# Primary: NVIDIA GLM-5.1
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "z-ai/glm-5.1"
NVIDIA_MAX_TOKENS = 300
NVIDIA_TEMPERATURE = 0.2
NVIDIA_TIMEOUT = 30
NVIDIA_MAX_RETRIES = 3
NVIDIA_RETRY_DELAY = 5

# Fallback 1: OpenRouter (Array for auto-fallback to avoid 429 errors)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS = [
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "inclusionai/ring-2.6-1t:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "arcee-ai/trinity-large-thinking:free",
    "openai/gpt-oss-120b:free",
    "z-ai/glm-4.5-air:free",
    "minimax/minimax-m2.5:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free"
]
OPENROUTER_MAX_TOKENS = 512
OPENROUTER_TEMPERATURE = 0.3
OPENROUTER_TIMEOUT = 45

# Fallback 2: Ollama local (any available model)
OLLAMA_FALLBACK_MODEL = "qwen3:0.6b"

# === Ollama (embeddings only) ===
OLLAMA_EMBEDDING_MODEL = "locusai/all-minilm-l6-v2"
EMBEDDING_DIMENSIONS = 384

# === Translation (Google Translate free API) ===
TRANSLATE_API_URL = "https://translate.googleapis.com/translate_a/single"
TRANSLATE_SOURCE_LANG = "en"   # Internal language: English
TRANSLATE_TARGET_LANG = "es"   # Web UI language: Spanish

# === CORS ===
CORS_ORIGINS = os.getenv("CPT_CORS_ORIGINS", "*")

# === Validation ===
RULE_MAX_LENGTH = 1000
PROHIBITED_TOKENS = ["require", "os", "io", "package", "dofile", "loadfile", "debug", "collectgarbage", "load"]

# === Learning Loop ===
LEARNING_MAX_ATTEMPTS = 5
LEARNING_RETRY_DELAY = 5
LEARNING_STEP_DELAY = 4  # 15 RPM (Under 20 RPM limit)
GOAL_THRESHOLD = 12.0

# === LoRA ===
LORA_RANK = 2
LORA_BASE_WEIGHTS = [1.0, 9.81, 0.1]  # speed, gravity, friction
LORA_LEARNING_RATE_FACTOR = 0.01

# === K-Means ===
KMEANS_N_CLUSTERS = 5
KMEANS_MAX_ITER = 100
