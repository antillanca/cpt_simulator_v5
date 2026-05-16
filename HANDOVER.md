# CPT Cognitive Engine v2.5 — Handover

> **Current Status (2026-05-15)**: 🚀 **MILESTONE ACHIEVED: 100% Curriculum Completion.**
> All 43 modules (Layer 0 to Layer 34) have been successfully assimilated, verified against symbolic invariants, and confirmed.

---

## 🏗️ Architecture: The Neuro-Symbolic Stack

CPT is a layered intelligence system that prioritizes **Determinism (Math)** over **Probability (LLMs)**.

### 1. Layer 0: Core Truth (Lua Sandbox)
- **Path**: `backend/core_truth/`
- **Logic**: Absolute source of truth. No AI. Just math and Lua 5.4.
- **Security**: Dockerized, no network, restricted resources.

### 2. Layer 1: Verifiers (Symbolic Invariants)
- **Path**: `backend/verifiers/`
- **Logic**: "The Arbiter". Verifies that any proposed rule (Lua or Neural) obeys physical laws (e.g., Energy Conservation, Non-Contradiction).

### 3. Layer 2: Intuition (Neural Tabular Filters)
- **Path**: `models/*.pt`
- **Logic**: Tiny PyTorch networks (TabularNet) trained on thousands of simulation samples.
- **Role**: Provides "Fast Thinking" (Intuition) for the A* Planner to prune impossible paths.

### 4. Layer 3: Reasoning (Student Engine)
- **Path**: `backend/ai/student_engine.py`
- **Logic**: Generative Lua code.
- **Role**: "Slow Thinking" (Reflective). Generates the actual physics rules that Layer 0 executes.

---

## 📈 Current Progress

| Category | Count | Status |
|:---|:---:|:---:|
| **Total Modules** | 43 | ✅ 100% |
| **Tabular Modules (.pt)** | 16 | ✅ Confirmed |
| **Lua Modules (Generated)** | 27 | ✅ Confirmed & Verified |
| **Failures/Rejected** | 0 | ✅ All repaired |

**Key Breakthrough**: We solved the "LLM Verbosity" problem using **Temperature 0.0** + **Aggressive Extraction Logic** in `_extract_lua()`, allowing complex modules like `magnetism_lorentz_force` and `quantum_double_slit_logic` to pass verification on the first attempt.

---

## 🛠️ Configuration & Tracking

- **Tracking**: Every Lua module now includes a `generated_by` field in `modules.json` to trace which model produced the logic.
- **Cascading Fallback**: `NVIDIA GLM-5.1` (Primary) → `OpenRouter` (Fallback) → `Ollama/Qwen3` (Last Resort).
- **Environment**: All keys (`OPENROUTER_API_KEY`, `NVIDIA_API_KEY`) are managed in `.env`.

---

## 🔮 Next Steps: Road to v3.0 (For LLM Brainstorming)

Now that the core curriculum is complete, the project moves into the **Distillation and Expansion** phase.

### 1. Knowledge Distillation (The "Oracle" Phase)
- **Task**: Use the confirmed Lua rules (Layer 3) as a **Teacher Oracle** to generate a massive DPO/GRPO dataset.
- **Goal**: Fine-tune a 1B-3B model (like Qwen2.5-Math or Llama-3.2-1B) so it inherits the "physics-correct" reasoning of the sandbox.

### 2. Curriculum Expansion
- **Electromagnetism**: Full Maxwell equations.
- **Chemistry**: Stoichiometry and atomic bonding logic.
- **Symbolic Logic**: Propositional and predicate calculus layers.

### 3. Edge Deployment
- Export the tiny neural filters (`.pt`) and the reasoning engine to **GGUF** format for execution on mobile/embedded devices without cloud access.

### 4. Tooling & UX
- **VS Code Plugin**: A developer tool that verifies code logic in real-time using the CPT Verifiers.
- **Hermes 2.0**: Enhance the agent to allow autonomous curriculum design (self-supervised learning).

---

## 📂 Key Files for Context
- `backend/core_truth/modules.json`: The complete brain state.
- `backend/ai/student_engine.py`: The generative reasoning logic.
- `scripts/training_orchestrator.py`: The master loop.
- `docs/ESTADO_CURRICULO.md`: Detailed module-by-module report.

---
**Verified by Antigravity AI.**
