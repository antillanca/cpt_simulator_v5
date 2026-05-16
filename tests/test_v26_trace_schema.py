from backend.traces.schema import ReasoningTrace, TraceStep, replay_trace


def test_trace_serialization_orders_by_step_id():
    trace = ReasoningTrace(
        steps=[
            TraceStep(
                step_id=2,
                rule="r",
                equation="c = 3",
                inputs={"before": {"x": 1}},
                operation="sandbox_execution",
                intermediate_result={"after": {"x": 3}},
                invariants_checked=["logic_basic"],
                verification={"passed": True, "violations": []},
                timestamp=2.0,
            ),
            TraceStep(
                step_id=1,
                rule="r",
                equation="b = 2",
                inputs={"before": {"x": 0}},
                operation="sandbox_execution",
                intermediate_result={"after": {"x": 1}},
                invariants_checked=["logic_basic"],
                verification={"passed": True, "violations": []},
                timestamp=1.0,
            ),
        ]
    )

    data = trace.to_dict()
    assert [step["step_id"] for step in data["steps"]] == [1, 2]


def test_trace_replay_returns_final_state():
    trace = ReasoningTrace(
        steps=[
            TraceStep(
                step_id=0,
                rule="r",
                equation="x = 1",
                inputs={"before": {"x": 0}},
                operation="sandbox_execution",
                intermediate_result={"after": {"x": 1}},
                invariants_checked=["logic_basic"],
                verification={"passed": True, "violations": []},
                timestamp=0.0,
            )
        ]
    )

    result = replay_trace(trace, initial_state={"x": 0})
    assert result.passed is True
    assert result.final_state["x"] == 1

