# FIN-C2-051 — Unit Tests: DocumentGenerationNode (BL-05 + TC-09/TC-10 extras)
#
# BL-05: Generated document is non-empty and contains an FSA check results section
# TC-09: validation_document / result non-empty on valid metrics
# TC-10: document_metadata pass/fail counts derived from fsa_checks
#
# DocumentGenerationNode.execute(state, config) reads performance_metrics,
# risk_metrics, fsa_checks (PASS/FAIL via the "result" key + "check_name"),
# model_metadata and validated_input.  It writes validation_document,
# llm_response, result (all identical) and document_metadata.


def _status_ok(result) -> bool:
    from framework.schemas.agent_status import AgentStatus

    return result.get("status") in (AgentStatus.SUCCESS, AgentStatus.SUCCESS.value)


# A synthetic connection string. The password part says "example" on purpose:
# the repository's own credential scan exempts values that announce themselves
# as fake, and a fixture that reads like a real credential should not be
# committed even as a probe.
_CONNECTION_URI = "postgresql://user:examplepw@host:5432/db"


def _status_err(result) -> bool:
    from framework.schemas.agent_status import AgentStatus

    return result.get("status") in (AgentStatus.ERROR, AgentStatus.ERROR.value)


def _performance_metrics():
    return {
        "sharpe_ratio": 1.42,
        "annualized_return": 0.18,
        "max_drawdown": 0.12,
        "calmar_ratio": 1.5,
        "sortino_ratio": 1.9,
        "win_rate": 0.62,
    }


def _risk_metrics():
    return {
        "VaR_95": 0.03,
        "CVaR_95": 0.05,
        "volatility": 0.14,
        "beta": 0.85,
        "drawdown_periods": [],
    }


def _fsa_checks(with_fail=True):
    checks = [
        {"check_id": "FSA-CHK-01", "check_name": "Sharpe Ratio", "result": "PASS", "detail": "Sharpe=1.42"},
        {"check_id": "FSA-CHK-02", "check_name": "Max Drawdown", "result": "PASS", "detail": "MaxDD=12%"},
        {"check_id": "FSA-CHK-03", "check_name": "Backtest Period", "result": "PASS", "detail": "36 months"},
    ]
    if with_fail:
        checks.append(
            {"check_id": "FSA-CHK-04", "check_name": "Out-of-sample Period", "result": "FAIL", "detail": "0 months"}
        )
    return checks


def _state(with_fail=True):
    return {
        "performance_metrics": _performance_metrics(),
        "risk_metrics": _risk_metrics(),
        "fsa_checks": _fsa_checks(with_fail=with_fail),
        "model_metadata": {
            "model_id": "MODEL-ALPHA",
            "version": "1.0.0",
            "asset_class": "equity",
            "strategy_type": "momentum",
        },
        "validated_input": {"period": "2024-01/2025-12", "validation_date": "2026-01-01"},
    }


class TestBL05DocumentContent:
    """BL-05: document is non-empty and contains an FSA check results section."""

    def test_document_non_empty(self):
        """validation_document must be a non-empty string."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state())

        assert _status_ok(result), f"Expected SUCCESS, got {result}. error_log: {result.get('error_log')}"
        doc = result.get("validation_document")
        assert isinstance(doc, str) and len(doc) > 0, "validation_document must be non-empty"

    def test_document_contains_fsa_check_section(self):
        """document must contain an FSA Regulatory Check Results section."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state())

        doc = result.get("validation_document", "")
        assert (
            "FSA Regulatory Check Results" in doc
        ), "document must contain the 'FSA Regulatory Check Results' section header"
        # The specific check IDs must be rendered into the document body.
        assert "FSA-CHK-01" in doc, "document must list FSA-CHK-01"
        assert "FSA-CHK-04" in doc, "document must list the failing FSA-CHK-04"

    def test_document_has_section_headers(self):
        """document must contain markdown '## ' section headers."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state())

        doc = result.get("validation_document", "")
        assert "## " in doc, f"document must contain '## ' headers, got: {doc[:120]!r}"


class TestTC09ValidationDocumentSurface:
    """TC-09: validation_document / result / llm_response are populated and identical."""

    def test_result_equals_validation_document(self):
        """result must equal validation_document (PostProcessNode reads result)."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state())

        assert result.get("result") == result.get(
            "validation_document"
        ), "result must be identical to validation_document for the output gate"

    def test_llm_response_populated(self):
        """llm_response must be populated (the downstream gate scans it)."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state())

        assert result.get("llm_response") == result.get(
            "validation_document"
        ), "llm_response must mirror validation_document for the credential scan"

    def test_zero_check_edge_case_no_crash(self):
        """Empty fsa_checks / metrics: document still generates (no crash)."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(
            {
                "performance_metrics": {},
                "risk_metrics": {},
                "fsa_checks": [],
                "model_metadata": {},
                "validated_input": {"period": "2024-01/2025-12"},
            }
        )

        assert _status_ok(result)
        assert result.get("validation_document"), "document must generate even with no checks"


class TestTC10DocumentMetadataCounts:
    """TC-10: document_metadata pass/fail counts are derived from fsa_checks."""

    def test_metadata_pass_fail_counts(self):
        """fsa_check_pass_count / fail_count must reflect the PASS/FAIL entries."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state(with_fail=True))

        metadata = result.get("document_metadata", {})
        assert metadata.get("fsa_check_pass_count") == 3, f"Expected 3 PASS, got {metadata.get('fsa_check_pass_count')}"
        assert metadata.get("fsa_check_fail_count") == 1, f"Expected 1 FAIL, got {metadata.get('fsa_check_fail_count')}"

    def test_metadata_model_id_and_dates(self):
        """document_metadata must carry model_id, generated_at, validation_date."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state())

        metadata = result.get("document_metadata", {})
        assert metadata.get("model_id") == "MODEL-ALPHA", "model_id must be set in metadata"
        assert "generated_at" in metadata, "generated_at must be present"
        assert "validation_date" in metadata, "validation_date must be present"

    def test_all_pass_yields_zero_fail_count(self):
        """A run with no FAIL checks must report fsa_check_fail_count = 0."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        node = DocumentGenerationNode()
        result = node.execute(_state(with_fail=False))

        metadata = result.get("document_metadata", {})
        assert metadata.get("fsa_check_fail_count") == 0
        assert metadata.get("fsa_check_pass_count") == 3


class TestTheDocumentGateWithholdsCompletely:
    """The document gate is exercised by calling execute() DIRECTLY.

    Direct calls are the point: the framework's own output gate sits in front
    of every node result and raises on the same credential shapes, so an
    end-to-end probe cannot tell which layer refused. Calling execute()
    directly is the only way to observe what THIS node returns.
    """

    def _tainted_state(self, taint):
        state = _state()
        state["model_metadata"] = dict(state["model_metadata"], strategy_type=f"momentum {taint}")
        return state

    def test_a_credential_in_the_assembled_document_withholds_every_field(self):
        from src.nodes.document_generation_node import (
            _OUTPUT_BEARING_FIELDS,
            DocumentGenerationNode,
        )

        result = DocumentGenerationNode().execute(self._tainted_state("AKIA1234567890ABCDEF"))

        assert _status_err(result), result
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result, (
                f"{field} omitted from the delta — LangGraph merges partial "
                "updates, so an omitted key keeps its previous value"
            )
            assert not result[field], f"{field} was not cleared: {result[field]!r}"

    def test_the_withheld_notice_is_truthy(self):
        """A falsy notice re-activates the framework's fallback onto result."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        result = DocumentGenerationNode().execute(self._tainted_state("sk_live_" + "abcdefghij0123456789"))
        assert result.get("formatted_output"), result.get("formatted_output")

    def test_the_message_names_the_field_and_never_the_value(self):
        from src.nodes.document_generation_node import DocumentGenerationNode

        taint = "AKIA1234567890ABCDEF"
        result = DocumentGenerationNode().execute(self._tainted_state(taint))
        rendered = repr(result)
        assert taint not in rendered
        assert "validation_document" in " ".join(result["error_log"])

    def test_an_ordinary_document_is_released(self):
        """Clean-path control: a refuse-everything gate cannot pass these tests."""
        from src.nodes.document_generation_node import DocumentGenerationNode

        result = DocumentGenerationNode().execute(_state())
        assert _status_ok(result), result
        assert "FSA Model Validation Document" in result["validation_document"]

    def test_prose_using_credential_vocabulary_is_still_released(self):
        """The gate must not fire on legitimate domain wording.

        The set this replaced was bare lowercase substrings, so any report
        mentioning a password or secret-rotation policy was refused outright.
        """
        from src.nodes.document_generation_node import DocumentGenerationNode

        state = _state()
        state["model_metadata"] = dict(state["model_metadata"], strategy_type="password_protected_secret_momentum")
        result = DocumentGenerationNode().execute(state)
        assert _status_ok(result), result

    def test_the_gate_delegates_to_the_framework_detector(self):
        """A local pattern set narrower than the framework's is a bypass.

        The framework raises inside its own output gate on anything it
        recognises, and the wrapper then discards this node's whole delta —
        the clearing included. So the local check must recognise at least
        everything the framework does.
        """
        from framework.security.credential_detector import detect_credentials

        from src.nodes.document_generation_node import _apply_output_gate

        for shape in (
            "sk_live_" + "abcdefghij0123456789",
            "sk-abcdefghijklmnopqrstuvwxyz",
            "AKIA1234567890ABCDEF",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqr",
            "Bearer abcdef1234567890abcdef",
            _CONNECTION_URI,
        ):
            assert detect_credentials(shape), f"probe shape is unknown to the framework: {shape}"
            assert _apply_output_gate(f"body {shape} body", {}) is not None

    def test_an_empty_document_is_withheld_rather_than_released(self):
        from src.nodes.document_generation_node import _apply_output_gate

        assert _apply_output_gate("   \n  ", {"model_id": "MODEL-ALPHA"}) is not None

    def test_a_credential_in_the_metadata_is_caught_too(self):
        """The gate covers document_metadata, not only the rendered text."""
        from src.nodes.document_generation_node import _apply_output_gate

        assert (
            _apply_output_gate(
                "a clean report body",
                {"model_id": "MODEL-ALPHA", "note": "AKIA1234567890ABCDEF"},
            )
            is not None
        )
