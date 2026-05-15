#!/usr/bin/env python3
"""Diagnóstico completo del learning loop - ejecución directa (sin API)."""
import sys, os, json, time, asyncio, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import (DEFAULT_STATE, LEARNING_MAX_ATTEMPTS,
                            LEARNING_RETRY_DELAY, LEARNING_STEP_DELAY, GOAL_THRESHOLD)
from backend.core.orchestrator import orchestrator
from backend.core.syllabus_manager import syllabus_manager
from backend.ai.tutor_engine import tutor_engine
from backend.memory.engine_lora import lora_adapter
from backend.persistence.database import SessionLocal, SyllabusItem

def db_stats():
    con = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.db"))
    cur = con.cursor()
    cur.execute("select count(*), count(case when success=1 then 1 end) from learning_logs")
    total, success = cur.fetchone()
    cur.execute("select count(*) from syllabus where is_completed=1")
    comp = cur.fetchone()[0]
    cur.execute("select error_message, count(*) as c from learning_logs where success=0 group by error_message order by c desc limit 5")
    errors = cur.fetchall()
    con.close()
    return total, success, comp, errors

import sqlite3

async def run_single_item(item, attempt_limit=5):
    """Correr un solo ítem del syllabus como lo hace learning_loop.py"""
    target_state = json.loads(item.target_state_json)
    last_failed_rule = None
    last_error = None
    lora_suggestions = lora_adapter.get_suggestions()

    for i in range(attempt_limit):
        # Generar/refinar regla
        if last_failed_rule and last_error:
            enhanced = f"{item.objective} [Adapted: gravity={lora_suggestions['gravity_value']:.2f}, friction={lora_suggestions['friction_coefficient']:.2f}]"
            rule_text = tutor_engine.refine_rule(last_failed_rule, last_error, enhanced)
        else:
            enhanced = f"{item.objective} [Use gravity={lora_suggestions['gravity_value']:.2f}, friction={lora_suggestions['friction_coefficient']:.2f}]"
            rule_text = tutor_engine.generate_rule(enhanced, orchestrator.current_state)

        if not rule_text:
            print(f"    Attempt {i+1}: No rule generated, skipping")
            syllabus_manager.log_attempt(item.id, "N/A", False, "No rule generated")
            await asyncio.sleep(LEARNING_RETRY_DELAY)
            continue

        print(f"    Attempt {i+1}: rule len={len(rule_text)}, first 80='{rule_text[:80]}...'")

        # Test en sandbox
        result = orchestrator.process_new_rule(rule_text)

        if result["status"] == "ok":
            actual = result["result"]["particle"]
            is_goal = True
            for key in target_state:
                if key in actual and abs(actual[key] - target_state[key]) > GOAL_THRESHOLD:
                    is_goal = False
                    break

            if is_goal:
                orchestrator.add_active_rule(rule_text)
                rule_id = orchestrator.save_rule(rule_text)
                syllabus_manager.log_attempt(item.id, rule_id, True)
                delta = np.array([
                    abs(actual.get("vx", 0)) / max(abs(lora_suggestions["speed_multiplier"]), 0.01),
                    abs(actual.get("vy", 0)) / max(abs(lora_suggestions["gravity_value"]), 0.01),
                    0.01,
                ])
                lora_adapter.adapt(feedback_score=1.0, delta_vector=delta)
                print(f"    ✅ SUCCESS! rule_id={rule_id}")
                return True
            else:
                last_error = f"Target not reached. Actual: {actual}, Target: {target_state}"
                last_failed_rule = rule_text
                syllabus_manager.log_attempt(item.id, "N/A", False, last_error)
                lora_adapter.adapt(feedback_score=-0.5, delta_vector=np.array([0.0, 0.0, 0.0]))
                print(f"    ❌ Goal not reached (attempt {i+1})")
        else:
            last_error = result.get("message", "unknown error")
            last_failed_rule = rule_text
            syllabus_manager.log_attempt(item.id, "N/A", False, last_error)
            lora_adapter.adapt(feedback_score=-1.0, delta_vector=np.array([0.0, 0.0, 0.0]))
            print(f"    ❌ Sandbox error: {last_error[:80]}")

        await asyncio.sleep(1)

    return False

async def main():
    t0 = time.time()
    banner = "=" * 70
    print(banner)
    print("  LEARNING LOOP DIRECT - DIAGNÓSTICO COMPLETO")
    print(banner)

    total_before, success_before, comp_before, errors_before = db_stats()
    print(f"\n📊 Estado DB ANTES de correr:")
    print(f"   learning_logs: {total_before} total, {success_before} success")
    print(f"   syllabus completados: {comp_before}/26")
    if errors_before:
        print(f"   Top errores: {errors_before}")

    # Reset estado
    orchestrator.current_state = dict(DEFAULT_STATE)
    orchestrator.active_rules = []
    lora_adapter.reset()

    # Verificar syllabus
    count = syllabus_manager.get_next_item()
    print(f"\n📋 Syllabus: próximo item = {count.title if count else 'Ninguno'}")

    if not count:
        print("   No hay items, re-seed...")
        subprocess.run([sys.executable, "seed_syllabus.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
        count = syllabus_manager.get_next_item()

    if not count:
        print("❌ No hay items después de seed!")
        return

    # Procesar hasta 3 items o hasta que se acaben
    processed = 0
    successes = 0
    for idx in range(3):
        item = syllabus_manager.get_next_item()
        if not item:
            print("\n✅ Todos los items completados!")
            break

        print(f"\n{'─'*60}")
        print(f"  Item {idx+1}: {item.title}")
        print(f"  Objective: {item.objective[:80]}...")
        print(f"  Target: {item.target_state_json}")

        try:
            success = await run_single_item(item, attempt_limit=3)
            if success:
                syllabus_manager.mark_completed(item.id)
                successes += 1
                print(f"  ✅ Marcado como completado: {item.title}")
            else:
                print(f"  ❌ No se completó: {item.title}")
        except Exception as e:
            print(f"  💥 EXCEPCIÓN: {e}")
            traceback.print_exc()

        processed += 1

    elapsed = time.time() - t0
    total_after, success_after, comp_after, errors_after = db_stats()

    # Detener cualquier loop activo
    try:
        import requests
        requests.post("http://localhost:8000/api/learning/stop", timeout=2, data=b'')
    except:
        pass

    print(f"\n{banner}")
    print(f"  RESULTADOS")
    print(banner)
    print(f"  Tiempo: {elapsed:.0f}s")
    print(f"  Items procesados: {processed}")
    print(f"  Éxitos (goal alcanzado): {successes}")
    print(f"")
    print(f"  learning_logs: {total_before} → {total_after} (+{total_after - total_before})")
    print(f"  learning_logs success: {success_before} → {success_after} (+{success_after - success_before})")
    print(f"  syllabus completados: {comp_before} → {comp_after} (+{comp_after - comp_before})")
    print(f"")

    if total_after > total_before:
        print(f"  📝 Nuevos logs:")
        con = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.db"))
        cur = con.cursor()
        cur.execute("select id, success, error_message, created_at from learning_logs order by id desc limit 10")
        for row in cur.fetchall():
            print(f"    id={row[0]} ok={row[1]} err='{row[2]}' time={row[3]}")
        con.close()

    print(banner)

if __name__ == "__main__":
    asyncio.run(main())