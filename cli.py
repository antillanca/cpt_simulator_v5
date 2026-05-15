#!/usr/bin/env python3
"""CPT Simulator v5 - CLI Agent Interface.

All output is in English (internal language).
Agents and automated systems use this interface for mathematical queries.
"""
import argparse
import json
import sys
import requests
from backend.notifier import notifier

API_BASE = "http://localhost:8000"


def cmd_status(args):
    """Get current mathematical state of the simulation."""
    try:
        resp = requests.get(f"{API_BASE}/api/state/math", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to CPT Simulator API. Is the server running?")
        sys.exit(1)


def cmd_test(args):
    """Test a Lua rule in the sandbox and return mathematical result."""
    rule_text = args.rule
    if not rule_text:
        print("ERROR: No rule text provided. Use: cli.py test '<lua_rule>'")
        sys.exit(1)

    try:
        resp = requests.post(
            f"{API_BASE}/api/rule/test",
            json={"rule_text": rule_text},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            print("=== Test Result ===")
            print(f"Raw state: {json.dumps(data.get('raw', {}), indent=2)}")
            print(f"Math: {json.dumps(data.get('math', {}), indent=2)}")
        else:
            try:
                error = resp.json()
                print(f"ERROR: {error.get('detail', resp.text)}")
            except:
                print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to CPT Simulator API. Is the server running?")
        sys.exit(1)


def cmd_learn_start(args):
    """Start the autonomous learning loop."""
    try:
        resp = requests.post(f"{API_BASE}/api/ai/learn/start", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Learning: {data.get('status', 'unknown')}")
        else:
            print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to CPT Simulator API. Is the server running?")
        sys.exit(1)


def cmd_learn_stop(args):
    """Stop the autonomous learning loop."""
    try:
        resp = requests.post(f"{API_BASE}/api/ai/learn/stop", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Learning: {data.get('status', 'unknown')}")
        else:
            print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to CPT Simulator API. Is the server running?")
        sys.exit(1)


def cmd_learn_status(args):
    """Get learning loop status."""
    try:
        resp = requests.get(f"{API_BASE}/api/ai/learn/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            status_str = "RUNNING" if data.get("is_running") else "IDLE"
            print(f"\n--- CPT COGNITIVE SYSTEM STATUS ---")
            print(f"Status:          {status_str}")
            print(f"Current Module:  {data.get('current_module', 'None')}")
            print(f"Cognitive Layer: {data.get('cognitive_layer', 'Unknown')}")
            print(f"Progress:        {data.get('confirmed_layers', 0)} confirmed / {data.get('pending', 0)} pending")
            print(f"Code Memory:     {data.get('memory_bytes', 0)} bytes")
            print(f"-----------------------------------\n")
        else:
            print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to CPT Simulator API. Is the server running?")
        sys.exit(1)


def cmd_syllabus_list(args):
    """List all syllabus items."""
    try:
        resp = requests.get(f"{API_BASE}/api/syllabus/list", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            if not items:
                print("No syllabus items found.")
            else:
                for item in items:
                    print(f"  [{item.get('order', '?')}] {item.get('title', 'N/A')} - {item.get('objective', 'N/A')}")
        else:
            print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to CPT Simulator API. Is the server running?")
        sys.exit(1)

def cmd_confirm(args):
    """Manually confirm a module with Lua code from file or stdin."""
    try:
        lua_code = args.lua_code
        if lua_code == "-":
            lua_code = sys.stdin.read()
        resp = requests.post(
            f"{API_BASE}/api/ai/confirm/{args.module_name}",
            json={"lua_code": lua_code},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            msg = f"✅ Module '{args.module_name}' confirmed ({data.get('code_size', 0)} bytes)."
            print(msg)
            notifier.send(f"📚 <b>Módulo Confirmado</b>: <i>{args.module_name}</i> ha sido asimilado por el StudentEngine.")
        else:
            print(f"ERROR: HTTP {resp.status_code} - {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to CPT Simulator API. Is the server running?")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CPT Simulator v5 - CLI Agent Interface (English output)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Get current mathematical state")

    # test
    test_parser = subparsers.add_parser("test", help="Test a Lua rule in sandbox")
    test_parser.add_argument("rule", type=str, help="Lua rule code to test")

    # learn-start
    subparsers.add_parser("learn-start", help="Start autonomous learning loop")

    # learn-stop
    subparsers.add_parser("learn-stop", help="Stop autonomous learning loop")

    # learn-status
    subparsers.add_parser("learn-status", help="Get learning loop status")

    # syllabus-list
    subparsers.add_parser("syllabus-list", help="List all syllabus items")

    # confirm
    confirm_parser = subparsers.add_parser("confirm", help="Manually confirm a module with Lua code")
    confirm_parser.add_argument("module_name", type=str, help="Module ID to confirm (e.g. layer_10_kinematics)")
    confirm_parser.add_argument("lua_code", type=str, help="Lua code string, or '-' to read from stdin")

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "test": cmd_test,
        "learn-start": cmd_learn_start,
        "learn-stop": cmd_learn_stop,
        "learn-status": cmd_learn_status,
        "syllabus-list": cmd_syllabus_list,
        "confirm": cmd_confirm,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
