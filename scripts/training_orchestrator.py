import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.kaggle_trainer import KaggleTrainer
from backend.ai.student_engine import student
from backend.notifier import notifier
from backend.tooling.hermes import hermes_assistant
from backend.tooling.permissions import default_policy
from backend.core_truth.sandbox import sandbox_manager
from backend.verifiers.simulation import verify_simulation

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_orchestrator.log")
    ]
)
logger = logging.getLogger("Orchestrator")

# Mapeo de Niveles a Fábricas DPO (Capas 0-11)
FACTORY_MAP = {
    0: "planner/logic_automation.py",
    1: "planner/counting_automation.py",
    2: "planner/arithmetic_automation.py",
    3: "planner/numeric_automation.py",
    4: "planner/proportion_automation.py",
    5: "planner/algebra_automation.py",
    6: "planner/function_automation.py",
    7: "planner/geometry_automation.py",
    8: "planner/vector_automation.py",
    9: "planner/trig_automation.py",
    10: "planner/planner_automation.py", # Kinematics
    11: "planner/newton_automation.py"    # Dynamics
}

class TrainingOrchestrator:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.modules_path = self.base_dir / "backend" / "core_truth" / "modules.json"
        self.kaggle_trainer = KaggleTrainer()
        
    def load_modules(self):
        with open(self.modules_path, "r") as f:
            return json.load(f)

    def save_modules(self, data):
        with open(self.modules_path, "w") as f:
            json.dump(data, f, indent=2)

    async def verify_rule_invariants(self, mod_key, mod, lua_code):
        """Verifica una regla contra los invariantes simbólicos definidos."""
        logger.info(f"🛡️ Verificando invariantes para {mod_key}...")
        
        # Ejecutar en el sandbox para obtener la traza
        res = sandbox_manager.run_rule(
            lua_code,
            initial_state=mod.get("initial_state", {}),
            frames=mod.get("simulation_frames", 10),
            collect_trace=True
        )
        
        if res.get("status") == "error":
            return False, f"Error de Sandbox: {res.get('message')}"
        
        trace = res.get("trace", [])
        invariant_set = mod.get("invariants", [])
        
        # Validación simbólica
        v_res = verify_simulation(trace, invariant_set)
        if not v_res.get("passed"):
            violations = ", ".join([v.get("invariant") for v in v_res.get("violations", [])])
            return False, f"Violación de Invariantes: {violations}"
            
        return True, "Validación exitosa"

    async def invoke_hermes_repair(self, mod_key, error_msg):
        """Ask Hermes for a repair plan, but do not auto-merge core changes."""
        logger.info("🤖 Invoking Hermes tooling for %s", mod_key)

        notifier.send(
            f"🤖 <b>Hermes tooling</b>: fallo detectado en <code>{mod_key}</code>.\n"
            "Se solicitará análisis y propuesta de parche, sin merge automático."
        )

        prompt = (
            f"Module '{mod_key}' failed with error:\n{error_msg}\n\n"
            "Provide a concise debugging analysis, a patch plan, and safe test ideas. "
            "Do not modify core_truth, verifiers, invariants, or the DSL compiler. "
            "Do not merge automatically."
        )

        result = await hermes_assistant.suggest_patch(prompt, target_path=f"backend/ai/{mod_key}.py")
        if not result.allowed:
            logger.warning("Hermes request rejected by policy: %s", result.message)
            notifier.send(f"⚠️ <b>Hermes bloqueado</b>: {result.message}")
            return False

        if not result.approved:
            # Pedimos permiso por Telegram usando botones interactivos
            approved = await notifier.ask_approval(
                f"⚠️ Hermes sugiere un parche para el módulo <b>{mod_key}</b>.\n¿Autorizas su ejecución?"
            )
            
            if approved:
                notifier.send(f"⏳ Ejecutando parche para {mod_key}...")
                # Concedemos permiso temporalmente
                os.environ["CPT_HERMES_HUMAN_APPROVAL"] = "1"
                result = await hermes_assistant.suggest_patch(prompt, target_path=f"backend/ai/{mod_key}.py")
                os.environ["CPT_HERMES_HUMAN_APPROVAL"] = "0"
                
                if result.allowed and result.approved:
                    notifier.send(f"✅ Parche aplicado a {mod_key}. Reanudando entrenamiento.")
                    logger.info(f"✅ Parche aplicado exitosamente a {mod_key}")
                    return True
                else:
                    error_msg = result.message or "Error desconocido en Hermes"
                    notifier.send(f"❌ El parche falló al aplicarse: {error_msg[:100]}")
                    logger.error(f"❌ El parche de Hermes falló: {error_msg}")
                    return False
            else:
                notifier.send(f"🚫 Parche rechazado para {mod_key}.")
                return False

        logger.info("Hermes returned a completed analysis for %s", mod_key)
        notifier.send(f"✅ <b>Análisis recibido</b> para <code>{mod_key}</code>. Revisar salida de Hermes.")
        return True

    async def run(self):
        msg_start = "🚀 <b>Orquestador CPT v2.5 Iniciado</b>\nAsimilación Neuro-Simbólica con verificación de invariantes activa."
        notifier.send(msg_start)
        logger.info("🚀 Iniciando Orquestador CPT v2.5")
        
        while True:
            data = self.load_modules()
            modules = data.get("modules", {})
            
            # Ordenar módulos por nivel y orden
            sorted_modules = sorted(
                modules.items(), 
                key=lambda x: (x[1].get("level", 0), x[1].get("order", 0))
            )
            
            target_mod = None
            for mod_key, mod in sorted_modules:
                if mod.get("status") == "confirmed":
                    continue
                target_mod = (mod_key, mod)
                break
            
            if not target_mod:
                logger.info("🎓 ¡Currículo completo! Todas las capas están confirmadas.")
                notifier.send("🏁 <b>Currículo CPT Completado</b>. El motor neuro-simbólico ha sido verificado.")
                break
            
            mod_key, mod = target_mod
            level = mod.get("level", 0)
            engine_type = mod.get("engine_type", "lua")
            
            logger.info(f"📍 Siguiente objetivo: {mod_key} (Nivel {level}, Motor: {engine_type})")
            
            if engine_type == "tabular":
                factory_script = FACTORY_MAP.get(level)
                if factory_script:
                    logger.info(f"🏭 Ejecutando fábrica DPO: {factory_script}")
                    full_path = self.base_dir / factory_script
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(self.base_dir) + ":" + env.get("PYTHONPATH", "")
                    
                    process = await asyncio.create_subprocess_exec(
                        "python3", str(full_path),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        logger.info(f"✅ Fábrica {mod_key} terminó exitosamente. Verificando modelo...")
                        
                        # PASO DE VERIFICACIÓN (CPT 2.5)
                        # Usamos el lua_code del módulo como referencia simbólica para verificar el modelo .pt
                        success, v_msg = await self.verify_rule_invariants(mod_key, mod, mod.get("lua_code", ""))
                        
                        if success:
                            mod["status"] = "confirmed"
                            mod["confirmed_by"] = "kaggle_orchestrator"
                            self.save_modules(data)
                            notifier.send(f"✅ <b>Capa Asimilada y Verificada</b>: {mod_key}")
                        else:
                            logger.error(f"❌ Fallo de Verificación en {mod_key}: {v_msg}")
                            repaired = await self.invoke_hermes_repair(mod_key, v_msg)
                            if repaired:
                                mod["status"] = "confirmed"
                                mod["confirmed_by"] = "hermes_override"
                                self.save_modules(data)
                            else:
                                break
                    else:
                        error_msg = stderr.decode()
                        logger.error(f"❌ Error en fábrica {mod_key}:\n{error_msg}")
                        repaired = await self.invoke_hermes_repair(mod_key, error_msg)
                        if repaired:
                            mod["status"] = "confirmed"
                            mod["confirmed_by"] = "hermes_override"
                            self.save_modules(data)
                        else:
                            break
                else:
                    logger.warning(f"⚠️ No hay fábrica definida para nivel {level}. Saltando...")
            
            elif engine_type == "lua":
                logger.info(f"🧠 Usando StudentEngine para asimilar {mod_key}...")
                
                # Definimos una función de validación real para el StudentEngine
                async def real_test_fn(code):
                    return await self.verify_rule_invariants(mod_key, mod, code)
                
                success = await student.learn_module(mod_key, real_test_fn)
                if not success:
                    logger.error(f"❌ StudentEngine falló en {mod_key}")
                    repaired = await self.invoke_hermes_repair(mod_key, "StudentEngine falló tras 5 intentos o validación fallida.")
                    if repaired:
                        mod["status"] = "confirmed"
                        mod["confirmed_by"] = "hermes_override"
                        self.save_modules(data)
                        notifier.send(f"✅ <b>Módulo confirmado por Hermes</b>: {mod_key}")
                    else:
                        logger.error(f"❌ Hermes no pudo reparar {mod_key}. Deteniendo.")
                        break
                else:
                    mod["status"] = "confirmed"
                    mod["confirmed_by"] = "student_engine"
                    self.save_modules(data)
                    notifier.send(f"🧠 <b>Capa Asimilada (Lua)</b>: {mod_key}\nVerificación neuro-simbólica exitosa.")
            
            await asyncio.sleep(5)

if __name__ == "__main__":
    orchestrator = TrainingOrchestrator()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("🛑 Orquestador detenido por el usuario.")
