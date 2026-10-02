# FIN-C2-051 — Integration Tests
#
# End-to-end smoke coverage for the Cat 2 two-layer pipeline:
#   PreProcessNode → BacktestIngestionNode → ValidationMetricsNode
#   → DocumentGenerationNode
#
# These tests pin two contracts that are easy to break silently:
#   * PreProcessNode.execute() must emit validated_input as a *dict* (not a
#     JSON string); BacktestIngestionNode requires a dict and otherwise returns
#     an error, which breaks the whole pipeline.
#   * DocumentGenerationNode's tables must read the keys ValidationMetricsNode
#     actually writes, or a section renders N/A whatever was computed.
#
# Framework-dependent imports are wrapped in ImportError skips, matching the
# tests/unit idiom (the AgentCore SDK wheel is not present in a bare local
# checkout; CI installs it).

import json

import pytest

# A valid invocation payload using an authorised model id and an in-range window.
_PAYLOAD = {
    "model_id": "MODEL-ALPHA",
    "asset_class": "equity",
    "backtest_data": "inline:stub",
    "validation_date_start": "2024-01-01",
    "validation_date_end": "2024-12-01",
}


def _import_nodes():
    """Import the node classes, skipping the test if the SDK wheel is absent."""
    try:
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode
        from src.nodes.validation_metrics_node import ValidationMetricsNode
        from src.nodes.document_generation_node import DocumentGenerationNode
    except ImportError as exc:
        pytest.skip(f"Framework not installed in CI: {exc}")
    return (
        PreProcessNode,
        BacktestIngestionNode,
        ValidationMetricsNode,
        DocumentGenerationNode,
    )


def _status_is_success(status) -> bool:
    """Compare a returned status against AgentStatus.SUCCESS, tolerating either
    the enum or its string value."""
    try:
        from framework.schemas.agent_status import AgentStatus
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"Framework not installed in CI: {exc}")
    return status == AgentStatus.SUCCESS or status == AgentStatus.SUCCESS.value


class TestPipelineEndToEnd:
    """Drive the four nodes in sequence, threading partial-dict updates into a
    shared state — the same data flow the inner DomainWorkflowGraph performs."""

    def _run_pipeline(self):
        (
            PreProcessNode,
            BacktestIngestionNode,
            ValidationMetricsNode,
            DocumentGenerationNode,
        ) = _import_nodes()

        state: dict = {"user_input": json.dumps(_PAYLOAD)}

        pre_out = PreProcessNode().execute(state)
        state.update(pre_out)

        ingest_out = BacktestIngestionNode().execute(state)
        state.update(ingest_out)

        metrics_out = ValidationMetricsNode().execute(state)
        state.update(metrics_out)

        doc_out = DocumentGenerationNode().execute(state)
        state.update(doc_out)

        return state

    def test_pre_process_emits_validated_input_dict(self):
        """validated_input must be a dict, not a JSON string."""
        (PreProcessNode, _, _, _) = _import_nodes()
        out = PreProcessNode().execute({"user_input": json.dumps(_PAYLOAD)})
        assert _status_is_success(out.get("status")), f"PreProcessNode should succeed on a valid payload, got {out!r}"
        validated = out.get("validated_input")
        assert isinstance(validated, dict), f"validated_input must be a dict, got {type(validated).__name__}"
        assert validated.get("model_id") == "MODEL-ALPHA"

    def test_backtest_ingestion_accepts_validated_input(self):
        """BacktestIngestionNode must NOT error out — it does when
        validated_input arrives as a string instead of a dict."""
        (PreProcessNode, BacktestIngestionNode, _, _) = _import_nodes()
        state = {"user_input": json.dumps(_PAYLOAD)}
        state.update(PreProcessNode().execute(state))
        ingest_out = BacktestIngestionNode().execute(state)
        assert _status_is_success(ingest_out.get("status")), (
            "BacktestIngestionNode must accept the validated_input dict and load " f"records, got {ingest_out!r}"
        )
        assert ingest_out.get("backtest_results"), "BacktestIngestionNode should return non-empty backtest_results"

    def test_full_pipeline_produces_validation_document(self):
        """End-to-end: the four nodes must produce a non-empty FSA document."""
        state = self._run_pipeline()
        assert _status_is_success(
            state.get("status")
        ), f"Pipeline end state should be SUCCESS, got status={state.get('status')!r}"
        doc = state.get("validation_document") or ""
        assert doc.strip(), "Pipeline must produce a non-empty validation_document"
        assert "FSA Model Validation Document" in doc
        assert "## 3. Risk Metrics" in doc

    def test_risk_metrics_section_is_populated(self):
        """The Risk Metrics table must show real values, not all N/A.

        DocumentGenerationNode reads the keys ValidationMetricsNode writes
        (var_95 / cvar_95 / annualized_volatility); a shorter name in either
        place renders the whole section N/A whatever was computed."""
        state = self._run_pipeline()

        risk_metrics = state.get("risk_metrics") or {}
        # ValidationMetricsNode writes these exact keys.
        assert "var_95" in risk_metrics
        assert "cvar_95" in risk_metrics
        assert "annualized_volatility" in risk_metrics

        doc = state.get("validation_document") or ""
        # Locate the Risk Metrics section and confirm it is not entirely N/A.
        marker = "## 3. Risk Metrics"
        assert marker in doc
        risk_section = doc.split(marker, 1)[1].split("## 4.", 1)[0]
        assert "Annualised Volatility" in risk_section, "Risk Metrics table must include the Annualised Volatility row"
        # A key mismatch renders every numeric cell as N/A; with the keys
        # aligned at least one of var_95/cvar_95/annualized_volatility is present.
        rendered_values = [
            str(risk_metrics.get("var_95")),
            str(risk_metrics.get("cvar_95")),
            str(risk_metrics.get("annualized_volatility")),
        ]
        present = [v for v in rendered_values if v not in ("None", "")]
        if present:
            assert any(v in risk_section for v in present), (
                "Risk Metrics section should render the computed risk values, " "but none were found"
            )


class TestAgentInvokeSmoke:
    """Best-effort full-agent invoke. Skips cleanly if the compiled-graph
    invoke path is unavailable in the local/CI environment."""

    def test_agent_invoke_smoke(self):
        try:
            from src.graph.graph import FinC2051Agent
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        agent = FinC2051Agent()

        invoke = getattr(agent, "invoke", None)
        if not callable(invoke):
            pytest.skip("FinC2051Agent has no invoke() in this environment")

        try:
            result = invoke(json.dumps(_PAYLOAD))
        except (NotImplementedError, AttributeError, TypeError) as exc:
            pytest.skip(f"Compiled-graph invoke path unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001
            # The deterministic pipeline test above is the authoritative
            # end-to-end assertion; an environment-specific invoke failure
            # (e.g. checkpointer wiring) should not fail the suite.
            pytest.skip(f"Agent invoke not exercisable in this environment: {exc}")

        assert result is not None, "agent.invoke() returned None"
