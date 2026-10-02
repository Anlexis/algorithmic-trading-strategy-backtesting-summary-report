# FIN-C2-051 — Proof of boundary: the error envelope carries no released content
#
# The framework projects an invocation's answer as
# `formatted_output or result`, with NO status check. Three consequences drive
# every assertion in this file:
#
#   1. A gate that returns an error WITHOUT clearing result still ships the
#      ungated answer inside the error envelope.
#   2. A falsy formatted_output ACTIVATES that fallback, so "" as a
#      withheld-output marker produces the exact leak it was written to prevent.
#   3. LangGraph merges partial deltas, so a key a node OMITS keeps whatever it
#      already held. Clearing therefore means returning the key with an empty
#      value — an assertion that only checks `not result.get(field)` passes on
#      a gate that clears nothing at all.
#
# The fault is injected on the DATA path — through the ingestion node's
# documented data-source adapter — never by patching a gate. Patching the gate
# would test the patch rather than the pipeline.

import json
import os

import pytest

_TOKEN = "containment-test-token"

_REQUEST = {
    "model_id": "MODEL-ALPHA",
    "asset_class": "equity",
    "backtest_data": "inline:reference",
    "validation_date_start": "2022-01-01",
    "validation_date_end": "2024-12-01",
}

# Shapes the framework's own credential detector recognises, plus the
# assignment form the template adds on top of it. A probe pattern the detector
# does NOT know would report the envelope as safe when it is not.
#
# The connection-string probe uses a password that says "example" on purpose:
# the repository's own credential scan exempts values that announce themselves
# as fake, and a fixture that reads like a real credential should not be
# committed even as a probe.
_CONNECTION_URI = "postgresql://user:examplepw@host:5432/db"

_CREDENTIAL_SHAPES = [
    "sk_live_" + "abcdefghij0123456789",
    "sk-abcdefghijklmnopqrstuvwxyz",
    "AKIA1234567890ABCDEF",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqr",
    "Bearer abcdef1234567890abcdef",
    _CONNECTION_URI,
    "password = hunter2supersecret",
]


def _tainted_client(monkeypatch, taint):
    """Build a client whose ingestion data source returns a tainted value."""
    from fastapi.testclient import TestClient

    import src.graph.domain_workflow_graph as domain
    import src.api.server as server
    from src.nodes.backtest_ingestion_node import (
        BacktestIngestionNode,
        _load_reference_data,
    )
    from src.graph.graph import Graph

    class TaintedSource:
        """The node's documented adapter, returning one tainted field."""

        def load(self, model_id, start_date, end_date, asset_class):
            data = _load_reference_data(model_id, start_date, end_date, asset_class)
            data["model_metadata"]["strategy_type"] = f"momentum {taint}"
            return data

    original = domain.DomainWorkflowGraph.register_nodes

    def register_with_tainted_source(self):
        original(self)
        self._nodes["backtest_ingestion"] = BacktestIngestionNode(data_source=TaintedSource())

    monkeypatch.setattr(domain.DomainWorkflowGraph, "register_nodes", register_with_tainted_source)
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    agent = Graph(config=server.runtime_config())
    agent.compile()
    monkeypatch.setattr(server, "agent", agent)
    return TestClient(server.app)


def _clean_client(monkeypatch):
    from fastapi.testclient import TestClient

    import src.api.server as server

    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    return TestClient(server.app)


def _invoke(client):
    return client.post(
        "/invoke",
        json={"input": json.dumps(_REQUEST), "session_id": "pb"},
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )


@pytest.fixture(autouse=True)
def _require_framework():
    try:
        import src.api.server  # noqa: F401
    except ImportError as exc:  # pragma: no cover - framework absent locally
        pytest.skip(f"Framework not installed: {exc}")


class TestErrorEnvelopeCarriesNothing:
    """Every credential shape reaches the caller as a refusal, never as content."""

    @pytest.mark.parametrize("taint", _CREDENTIAL_SHAPES)
    def test_the_envelope_contains_neither_the_taint_nor_the_document(self, monkeypatch, taint):
        response = _invoke(_tainted_client(monkeypatch, taint))
        assert response.status_code == 200
        body = response.json()
        envelope = json.dumps(body, default=str)

        assert body["status"] == "error", body
        assert taint not in envelope, "the tainted value reached the caller"
        # Nothing from the document body survives either — the report renders
        # the model identifier in its title and in every section header.
        assert "FSA Model Validation Document" not in envelope
        assert "Executive Summary" not in envelope

    @pytest.mark.parametrize("taint", _CREDENTIAL_SHAPES)
    def test_the_envelope_leaks_no_traceback_and_no_source_path(self, monkeypatch, taint):
        body = _invoke(_tainted_client(monkeypatch, taint)).json()
        envelope = json.dumps(body, default=str)
        assert "Traceback" not in envelope
        assert "site-packages" not in envelope
        assert os.sep + "Users" + os.sep not in envelope
        assert '.py", line' not in envelope

    def test_the_withheld_notice_is_truthy_where_the_gate_answers(self, monkeypatch):
        """The assignment shape reaches the response gate, which answers with a notice.

        A falsy notice here would re-activate the framework's fallback onto
        `result`, so the notice being non-empty is the assertion that matters —
        not merely that the status is an error.
        """
        from src.nodes.post_process_node import _WITHHELD_NOTICE

        body = _invoke(_tainted_client(monkeypatch, "password = hunter2supersecret")).json()
        assert body["status"] == "error"
        assert body["output"] == _WITHHELD_NOTICE
        assert body["output"], "a falsy notice re-activates the result fallback"
        assert "PostProcessNode" in body["node_history"], body["node_history"]


class TestCleanPathControl:
    """A refuse-everything gate must not be able to pass the tests above."""

    def test_the_same_request_still_returns_its_real_answer(self, monkeypatch):
        response = _invoke(_clean_client(monkeypatch))
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success", body
        assert "FSA Model Validation Document" in body["output"]
        assert "MODEL-ALPHA" in body["output"]

    def test_the_clean_answer_passes_through_the_output_gate(self, monkeypatch):
        body = _invoke(_clean_client(monkeypatch)).json()
        assert "PostProcessNode" in body["node_history"], "the answer must be released BY the gate, not around it"


class TestWithheldDeltasClearRatherThanOmit:
    """Clearing is asserted as presence AND emptiness, at both gates.

    `assert not delta.get(field)` would be satisfied by a gate that returns no
    such key at all — and LangGraph would then leave the old value in state.
    """

    def test_the_response_gate_clears_every_output_field(self):
        from src.nodes.post_process_node import (
            _OUTPUT_BEARING_FIELDS,
            PostProcessNode,
        )

        delta = PostProcessNode().execute(
            {
                "result": "leaked AKIA1234567890ABCDEF inside the report",
                "validation_document": "the full report",
                "llm_response": "the full report",
                "document_metadata": {"model_id": "MODEL-ALPHA"},
            }
        )
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in delta, f"{field} omitted — the old value would survive"
            assert not delta[field], f"{field} not cleared: {delta[field]!r}"
        assert delta["formatted_output"]

    def test_the_document_gate_clears_every_output_field(self):
        from src.nodes.document_generation_node import (
            _OUTPUT_BEARING_FIELDS,
            _withheld_delta,
        )

        delta = _withheld_delta("'validation_document' contains a credential-shaped value")
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in delta, f"{field} omitted — the old value would survive"
            assert not delta[field], f"{field} not cleared: {delta[field]!r}"
        assert delta["formatted_output"]

    def test_no_violation_message_carries_the_matched_value(self):
        """Messages name a field or a pattern class, never the match.

        The framework's mandatory gate scans every value a node returns, so an
        echoed credential would make it raise and DISCARD the whole delta —
        the clearing included.
        """
        from src.nodes.post_process_node import _withheld_delta

        secret = "AKIA1234567890ABCDEF"
        delta = _withheld_delta("'result' matched the aws_key pattern class")
        assert secret not in json.dumps(delta, default=str)


class TestTheGateUsesTheFrameworkDetector:
    """A local pattern set narrower than the framework's is a containment bypass.

    A value the framework catches and the template misses makes the framework
    raise inside its own output gate, and the wrapper then returns a bare error
    partial that discards the node's whole delta — this gate's clearing with it.
    """

    @pytest.mark.parametrize("shape", _CREDENTIAL_SHAPES[:-1])
    def test_every_framework_shape_is_also_caught_locally(self, shape):
        from src.nodes.post_process_node import _apply_output_gate

        assert _apply_output_gate(f"Report body containing {shape} in prose")

    def test_the_local_set_adds_the_assignment_form(self):
        """The framework detector does not carry the assignment shape."""
        from framework.security.credential_detector import detect_credentials

        from src.nodes.post_process_node import _apply_output_gate

        assignment = "password = hunter2supersecret"
        assert not detect_credentials(assignment)
        assert _apply_output_gate(assignment) == "credential_assignment"

    def test_an_ordinary_validation_document_is_not_refused(self):
        """The gate must not fire on legitimate domain prose.

        The set this replaced was bare lowercase substrings, so any report that
        merely used the word "secret" or "password" was refused outright.
        """
        from src.nodes.post_process_node import _apply_output_gate

        prose = (
            "# FSA Model Validation Document\n\n"
            "The model registry password policy and the secret-rotation "
            "schedule are documented separately. No api_key material is held "
            "by this model. Bearer of record: the model risk committee.\n"
        )
        assert _apply_output_gate(prose) is None
