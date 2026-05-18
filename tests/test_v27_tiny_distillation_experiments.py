import json
from pathlib import Path

import pytest

from backend.neural.tiny_experiments import TrainConfig, load_checkpoint, train_model
from backend.validation.failure_taxonomy import classify_failure
from backend.validation.model_eval import invariant_retention_rate
from backend.validation.oracle_arena import ArenaExample, ArenaResult, compare_oracle_vs_model
from scripts.generate_large_dataset import GenerationConfig, generate_large_dataset
from scripts.model_eval_runner import evaluate_dataset


def _write_modules_file(tmp_path: Path) -> Path:
    payload = {
        "modules": {
            "layer_00": {
                "level": 0,
                "order": 0,
                "description": "layer zero",
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
                "target_state": {"x": 1},
            },
            "layer_01": {
                "level": 1,
                "order": 0,
                "description": "layer one",
                "simulation_frames": 1,
                "invariants": ["logic_basic"],
                "target_state": {"x": 2},
            },
        }
    }
    path = tmp_path / "modules.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _patch_sandbox(monkeypatch, module):
    def fake_run_rule(rule_text, initial_state=None, timeout_ms=None, frames=1, collect_trace=False):
        initial_state = dict(initial_state or {})
        final = dict(initial_state)
        final["x"] = final.get("x", 0) + 1
        return {
            "status": "ok",
            "particle": final,
            "trace": {
                "steps": [
                    {
                        "before": initial_state,
                        "after": final,
                        "frame": 0,
                    }
                ]
            },
        }

    monkeypatch.setattr(module.sandbox_manager, "run_rule", fake_run_rule)
    monkeypatch.setattr(module, "verify_simulation", lambda trace, invariants: {"passed": True, "violations": []})


def test_large_dataset_generation_deterministic_and_resumable(tmp_path, monkeypatch):
    import scripts.generate_large_dataset as gen_mod

    modules_path = _write_modules_file(tmp_path)
    _patch_sandbox(monkeypatch, gen_mod)

    out1 = tmp_path / "run1"
    cfg1 = GenerationConfig(seed=7, output_dir=out1, num_samples=6, shard_size=2, layer_weights={0: 3.0, 1: 1.0}, modules_path=modules_path)
    out_dir = generate_large_dataset(cfg1)
    assert out_dir == out1
    assert (out1 / "dataset.manifest.json").exists()
    assert (out1 / "shard_manifest.json").exists()
    assert len(list(out1.glob("shard_*.jsonl"))) == 3

    rows = []
    for shard in sorted(out1.glob("shard_*.jsonl")):
        rows.extend([json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()])
    assert len(rows) == 6
    assert sum(1 for row in rows if row["curriculum_layer"] == 0) > sum(1 for row in rows if row["curriculum_layer"] == 1)

    out2 = tmp_path / "run2"
    cfg2a = GenerationConfig(seed=7, output_dir=out2, num_samples=4, shard_size=2, layer_weights={0: 3.0, 1: 1.0}, modules_path=modules_path)
    generate_large_dataset(cfg2a)
    cfg2b = GenerationConfig(seed=7, output_dir=out2, num_samples=6, shard_size=2, layer_weights={0: 3.0, 1: 1.0}, modules_path=modules_path, resume=True)
    generate_large_dataset(cfg2b)

    rows_resume = []
    for shard in sorted(out2.glob("shard_*.jsonl")):
        rows_resume.extend([json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()])
    assert [row["sample_id"] for row in rows_resume] == [row["sample_id"] for row in rows]


def test_arena_and_failure_taxonomy():
    oracle = {
        "sample_id": "s1",
        "question": "Q",
        "structured_state": {"initial_state": {"x": 0}},
        "reasoning_trace": [{"step_id": 0, "rule": "r", "equation": "x=1", "inputs": {"before": {"x": 0}}, "operation": "sandbox_execution", "intermediate_result": {"after": {"x": 1}}, "invariants_checked": ["logic_basic"], "verification": {"passed": True, "violations": []}, "timestamp": 1.0}],
        "final_answer": {"x": 1},
        "verification_status": {"passed": True, "violations": []},
        "module_source": "m::layer_00",
        "curriculum_layer": 0,
    }
    model = dict(oracle)
    example = ArenaExample(sample_id="s1", question="Q", oracle=oracle, model_output=model, metadata={"initial_state": {"x": 0}})
    result = compare_oracle_vs_model(example)
    assert result.exact_match is True
    assert result.replay_consistency is True
    assert classify_failure(result) is None

    failing = ArenaResult(
        exact_match=False,
        struct_match=False,
        invariant_violation=True,
        trace_consistency=0.0,
        answer_consistency=0.0,
        trajectory_deviation=1.0,
        replay_consistency=False,
        sample_id="x",
        module_source="m",
        curriculum_layer=0,
    )
    assert classify_failure(failing) == "replay_instability"


def test_invariant_retention_metric():
    preds = [{"verification_status": {"passed": True}, "invariants_checked": ["logic_basic"]}]
    oracles = [{"invariants_checked": ["logic_basic"]}]
    assert invariant_retention_rate(preds, oracles) == 1.0


def test_training_and_evaluation_reproducibility(tmp_path):
    rows = []
    for i in range(6):
        rows.append(
            {
                "sample_id": f"s{i}",
                "question": f"Q{i}",
                "structured_state": {"initial_state": {"x": i}, "parameters": {}, "module": "m", "module_version": "v"},
                "reasoning_trace": [],
                "equations_used": [],
                "invariants_checked": ["logic_basic"],
                "final_answer": {"x": i},
                "verification_status": {"passed": True, "violations": []},
                "module_source": "m::layer_00",
                "curriculum_layer": 0,
                "seed": 7,
                "timestamp": float(i),
                "dataset_version": "2.7.0",
                "snapshot_hash": "snap",
                "module_hash": "mod",
            }
        )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    cfg = TrainConfig(model_type="transformer", seed=5, data_path=dataset, output_dir=tmp_path / "train_out", epochs=1, batch_size=2, lr=1e-3, max_steps=1, device="cpu", eval_every=1, save_every=1)
    train_result = train_model(cfg)
    checkpoint = Path(train_result["checkpoint_path"])
    assert checkpoint.exists()

    model, tokenizer, payload = load_checkpoint(checkpoint)
    assert payload["model_type"] == "transformer"
    assert tokenizer.pad_id >= 0

    report1 = evaluate_dataset(dataset, checkpoint=checkpoint, model_type="transformer", seed=5, output_path=tmp_path / "eval1.json")
    report2 = evaluate_dataset(dataset, checkpoint=checkpoint, model_type="transformer", seed=5, output_path=tmp_path / "eval2.json")
    assert report1["evaluation"]["fingerprint"] == report2["evaluation"]["fingerprint"]
    assert report1["arena"]["by_module"] == report2["arena"]["by_module"]

