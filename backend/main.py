"""CPT Simulator v5 - FastAPI Gateway.

All internal text is English. Web-facing responses are translated to Spanish via i18n.
CLI/agent responses remain in English.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core.orchestrator import orchestrator
from backend.persistence.database import init_db
from backend.core.syllabus_manager import syllabus_manager
# LEARNING LOOP VIEJO DESACTIVADO - usar /api/ai/learn/start en su lugar
# from backend.core.learning_loop import learning_loop
learning_loop = None  # Placeholder para backward compat
from backend.core.math_formatter import math_formatter
from backend.i18n import translate, translate_dict
from backend.config import CORS_ORIGINS, DEFAULT_STATE

# === AI Learning System ===
from backend.ai.learning_orchestrator import orchestrator as ai_orchestrator
import time

import json
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle (replaces deprecated @app.on_event)."""
    # Startup
    init_db()
    print("[CPT] Database initialized.")
    print("[CPT] Simulator ready.")
    yield
    # Shutdown: stop AI learning loop gracefully
    if ai_orchestrator.is_running:
        ai_orchestrator.stop()
        print("[CPT] AI learning loop stopped on shutdown.")
    # Old learning_loop is None (disabled), but check for safety
    try:
        if learning_loop is not None and learning_loop.is_running:
            learning_loop.stop()
            print("[CPT] Old learning loop stopped on shutdown.")
    except Exception:
        pass


app = FastAPI(title="CPT Simulator v5 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(",") if CORS_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RuleRequest(BaseModel):
    rule_text: str

class SyllabusItemRequest(BaseModel):
    title: str
    objective: str
    target_state: dict
    order: int


# === State Endpoints ===

@app.get("/api/state/math")
async def get_math_state():
    """Mathematical state representation (English, for CLI/agents)."""
    state = orchestrator.current_state
    return math_formatter.format_state(state)


@app.get("/api/state/ui")
async def get_ui_state():
    """State for web UI (translated to Spanish)."""
    state = orchestrator.current_state
    math = math_formatter.format_state(state)
    return await translate_dict(math)


# === Rule Endpoints (non-blocking via asyncio.to_thread) ===

@app.post("/api/rule/test")
async def test_rule(request: RuleRequest):
    """Test a Lua rule in the sandbox. Non-blocking."""
    result = await asyncio.to_thread(orchestrator.process_new_rule, request.rule_text)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    math_res = math_formatter.format_state(result["result"]["particle"])
    return {"status": "ok", "raw": result["result"], "math": math_res}


@app.post("/api/rule/simulate")
async def simulate_rule(request: RuleRequest):
    """Pure simulation: test rule WITHOUT mutating state. Non-blocking."""
    result = await asyncio.to_thread(orchestrator.simulate_rule, request.rule_text)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    math_res = math_formatter.format_state(result["result"]["particle"])
    return {"status": "ok", "raw": result["result"], "math": math_res}


@app.post("/api/rule/save")
async def save_rule(request: RuleRequest):
    """Save and activate a rule. Non-blocking."""
    result = await asyncio.to_thread(orchestrator.commit_rule, request.rule_text)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    orchestrator.add_active_rule(request.rule_text)
    rule_id = await asyncio.to_thread(orchestrator.save_rule, request.rule_text)
    return {"status": "ok", "rule_id": rule_id, "message": "Rule saved and activated"}


# === Syllabus Endpoints ===

@app.post("/api/syllabus/add")
async def add_syllabus_item(request: SyllabusItemRequest):
    """Add a syllabus item (internal: English, UI response: Spanish)."""
    syllabus_manager.add_item(
        request.title,
        request.objective,
        request.target_state,
        request.order,
    )
    return {"status": "ok", "message": await translate("Syllabus item added successfully.")}


@app.get("/api/syllabus/list")
async def list_syllabus():
    """List all syllabus items (English for agents)."""
    items = syllabus_manager.list_items()
    return {"items": items}


@app.get("/api/syllabus/list/ui")
async def list_syllabus_ui():
    """List syllabus items for web UI (Spanish)."""
    items = syllabus_manager.list_items()
    return {"items": [await translate_dict(i) if isinstance(i, dict) else i for i in items]}


# === Learning Loop Endpoints (Deprecated - use /api/ai/learn/*) ===

@app.post("/api/learning/start")
async def start_learning():
    """DEPRECATED: Use /api/ai/learn/start instead."""
    raise HTTPException(status_code=410, detail="Deprecated. Use /api/ai/learn/start")


@app.post("/api/learning/stop")
async def stop_learning():
    """DEPRECATED: Use /api/ai/learn/stop instead."""
    raise HTTPException(status_code=410, detail="Deprecated. Use /api/ai/learn/stop")


@app.get("/api/learning/status")
async def learning_status():
    """DEPRECATED: Use /api/ai/learn/status instead."""
    raise HTTPException(status_code=410, detail="Deprecated. Use /api/ai/learn/status")


# === WebSocket ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time simulation state broadcast."""
    await websocket.accept()
    try:
        while True:
            state = orchestrator.step()
            await websocket.send_json({
                "type": "state",
                "data": state,
            })
            await asyncio.sleep(0.016)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.0001)
                message = json.loads(data)
                if message.get("type") == "pause":
                    orchestrator.is_paused = True
                elif message.get("type") == "resume":
                    orchestrator.is_paused = False
                elif message.get("type") == "reset":
                    orchestrator.current_state = dict(DEFAULT_STATE)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        print("[CPT] WebSocket client disconnected.")
    except Exception as e:
        print(f"[CPT] WebSocket error: {e}")


from backend.ai.agent_observer import agent_i_at


# === AI Learning System Endpoints ===

@app.post("/api/ai/chat")
async def ai_chat(request: dict):
    """Chat with the i@ Observer Agent (Natural Language)."""
    user_text = request.get("text", "")
    if not user_text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Process through i@ Agent
    result = await agent_i_at.process_request(user_text)
    
    # Translate the message part for the user
    if "message" in result:
        result["message"] = await translate(result["message"])
        
    return result


@app.post("/api/ai/learn/start")
async def ai_learn_start():
    """Start the AI learning loop (student + teacher)."""
    if ai_orchestrator.is_running:
        return {"status": "already_running"}
    # Run as async task (not thread)
    asyncio.create_task(ai_orchestrator.start())
    return {"status": "started", "message": "AI learning loop started"}


@app.post("/api/ai/learn/stop")
async def ai_learn_stop():
    """Stop the AI learning loop."""
    ai_orchestrator.stop()
    return {"status": "ok", "message": await translate("AI learning loop stopping...")}


@app.post("/api/ai/dpo/start")
async def ai_dpo_start(background_tasks: BackgroundTasks):
    """Trigger background DPO dataset generation."""
    from scripts.run_dpo_factory import main as run_dpo
    background_tasks.add_task(run_dpo)
    return {"status": "ok", "message": await translate("DPO generation started in background.")}


@app.get("/api/ai/learn/status")
async def ai_learn_status():
    """Get current AI learning status."""
    return ai_orchestrator.get_status()


@app.get("/api/ai/monitor")
async def ai_monitor():
    """Detailed monitor for MoE Training."""
    import os
    import subprocess
    from backend.ai.student_engine import EXPERT_MAPPING
    
    # Check if factory is running
    factory_running = False
    try:
        # Avoid pgrep failing on some systems, use ps
        res = subprocess.run(["pgrep", "-f", "run_dpo_factory.py"], capture_output=True)
        factory_running = res.returncode == 0
    except:
        pass

    # Check dataset sizes
    datasets = {}
    experts = set(EXPERT_MAPPING.values())
    for expert in experts:
        path = f"dpo_dataset_{expert}.jsonl"
        if os.path.exists(path):
            try:
                datasets[expert] = {
                    "size_kb": round(os.path.getsize(path) / 1024, 2),
                    "lines": sum(1 for _ in open(path, "rb")),
                }
            except:
                datasets[expert] = {"size_kb": 0, "lines": 0}
        else:
            datasets[expert] = {"size_kb": 0, "lines": 0}
            
    # Tail log
    logs = []
    if os.path.exists("dpo_factory.log"):
        try:
            with open("dpo_factory.log", "r", errors="ignore") as f:
                logs = f.readlines()[-30:]
        except:
            pass
            
    return {
        "factory_running": factory_running,
        "datasets": datasets,
        "logs": [l.strip() for l in logs],
        "experts_count": len(experts),
    }


@app.get("/api/ai/modules")
async def ai_list_modules():
    """List all modules (layers) and their status."""
    return ai_orchestrator.student.modules


@app.get("/api/ai/modules/{module_name}")
async def ai_get_module(module_name: str):
    """Get a specific module."""
    mod = ai_orchestrator.student.get_module(module_name)
    if not mod:
        raise HTTPException(status_code=404, detail="Module not found")
    return mod


@app.get("/api/ai/layers")
async def ai_get_layers():
    """Get all confirmed layers (compiled knowledge)."""
    layers = ai_orchestrator.student.get_confirmed_layers()
    return {
        "layers": [
            {
                "name": name,
                "level": mod["level"],
                "subject": mod.get("subject", ""),
                "description": mod["description"],
                "code_size": len(mod.get("lua_code", "") or ""),
                "confirmed_at": mod.get("confirmed_at"),
            }
            for name, mod in layers
        ],
        "total_code": ai_orchestrator.student.get_active_lua(),
    }


@app.get("/api/ai/memory")
async def ai_memory():
    """Report memory usage of the AI learning system."""
    return ai_orchestrator.student.memory_usage()


@app.get("/api/ai/observer")
async def ai_observer():
    """Get Luenberger observer progress report."""
    return ai_orchestrator.teacher.get_observer_report()


@app.post("/api/ai/reset")
async def ai_reset():
    """Reset rejected modules to pending and restart learning."""
    from backend.ai.student_engine import student, load_modules, save_modules
    # Reload modules from disk
    student.modules = load_modules()
    # Reset rejected to pending
    reset_count = 0
    for name, mod in student.modules.get("modules", {}).items():
        if mod.get("status") == "rejected":
            mod["status"] = "pending"
            mod.pop("rejection_reason", None)
            mod["rejection_count"] = 0
            reset_count += 1
    save_modules(student.modules)
    return {"status": "reset", "modules_reset": reset_count}


@app.post("/api/ai/confirm/{module_name}")
async def ai_confirm(module_name: str, request: dict):
    """Manually confirm a module with provided Lua code."""
    from backend.ai.student_engine import student, save_modules
    lua_code = request.get("lua_code", "")
    if lua_code and module_name in student.modules.get("modules", {}):
        student.modules["modules"][module_name]["status"] = "confirmed"
        student.modules["modules"][module_name]["lua_code"] = lua_code
        student.modules["modules"][module_name]["confirmed_by"] = "owl-alpha"
        student.modules["modules"][module_name]["confirmed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_modules(student.modules)
        return {"status": "confirmed", "module": module_name, "code_size": len(lua_code)}
    return {"status": "error", "message": "Invalid module or code"}


# ─── NEW: Monitoring Endpoints for the Cognitive Dashboard ──────────────────

@app.get("/api/ai/curriculum/status")
async def curriculum_status():
    """Return real-time curriculum statistics for the frontend monitor."""
    import json
    from pathlib import Path
    modules_path = Path("backend/core_truth/modules.json")
    if not modules_path.exists():
        raise HTTPException(status_code=404, detail="modules.json not found")
    with open(modules_path) as f:
        data = json.load(f)
    modules = data.get("modules", {})

    total = len(modules)
    confirmed = sum(1 for m in modules.values() if m.get("status") == "confirmed")
    pending   = sum(1 for m in modules.values() if m.get("status") == "pending")
    repair    = sum(
        1 for m in modules.values()
        if m.get("status") == "confirmed" and
           (not m.get("lua_code", "").strip() or m.get("rejection_count", 0) > 0)
    )
    tabular   = sum(1 for m in modules.values() if m.get("engine_type") == "tabular")
    lua_count = sum(1 for m in modules.values() if m.get("engine_type") == "lua")

    modules_list = sorted([
        {"key": k, "level": v.get("level", 0), "subject": v.get("subject", ""),
         "status": v.get("status", ""), "engine_type": v.get("engine_type", "lua")}
        for k, v in modules.items()
    ], key=lambda x: x["level"])

    return {
        "total": total, "confirmed": confirmed, "pending": pending,
        "repair": repair, "tabular": tabular, "lua": lua_count,
        "modules": modules_list
    }


@app.get("/api/ai/orchestrator/log")
async def orchestrator_log():
    """Return the last N lines of the orchestrator log and current execution state."""
    from pathlib import Path
    log_path = Path("training_orchestrator.log")
    lines = []
    if log_path.exists():
        with open(log_path, "r") as f:
            lines = [l.rstrip() for l in f.readlines()[-80:]]

    return {
        "lines": lines,
        "current_module": None,   # TODO: persist current_module from orchestrator to a status file
        "engine_type": None,
        "kaggle_running": False,  # TODO: check Kaggle kernel status file
        "student_running": False
    }


@app.get("/api/ai/models")
async def list_models():
    """Return the list of .pt model files currently loaded in models/."""
    from pathlib import Path
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}
    models = [f.name for f in sorted(models_dir.glob("*.pt"))]
    return {"models": models}
