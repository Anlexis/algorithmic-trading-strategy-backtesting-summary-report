# FIN-C2-051 — Integration tests: the deployed /invoke contract
#
# These drive the REAL ASGI application, not the node functions, because every
# defect they pin is invisible at node level:
#
#   * the trust gate runs inside the framework's node wrapper, which a direct
#     execute() call bypasses entirely;
#   * the validated payload only reaches the inner domain graph through the
#     outer→inner bridge, which only exists during a real invocation;
#   * a value declared in config/config.yaml only reaches the node that reads
#     it if the entry point actually loads the file and passes it down.
#
# Every case here therefore goes through the HTTP entry point with Bearer auth.

import json

import pytest

_TOKEN = "integration-test-token"

# A synthetic connection string. The password part says "example" on purpose:
# the repository's own credential scan exempts values that announce themselves
# as fake, and a fixture that reads like a real credential should not be
# committed even as a probe.
_CONNECTION_URI = "postgresql://user:examplepw@host:5432/db"

_BASE_REQUEST = {
    "model_id": "MODEL-ALPHA",
    "asset_class": "equity",
    "backtest_data": "inline:reference",
    "validation_date_start": "2022-01-01",
    "validation_date_end": "2024-12-01",
}

_RECORDS = [
    {"date": "2022-01-01", "portfolio_return": 0.01, "benchmark_return": 0.008},
    {"date": "2022-06-01", "portfolio_return": 0.02, "benchmark_return": 0.011},
    {"date": "2023-01-01", "portfolio_return": -0.03, "benchmark_return": 0.004},
    {"date": "2024-12-01", "portfolio_return": 0.015, "benchmark_return": 0.009},
]


@pytest.fixture()
def client(monkeypatch):
    """A test client over the real application, with caller auth configured."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
    try:
        from fastapi.testclient import TestClient

        import src.api.server as server
    except ImportError as exc:  # pragma: no cover - framework absent locally
        pytest.skip(f"Framework not installed: {exc}")
    return TestClient(server.app)


def _post(client, *, request=None, context=None, token=_TOKEN, raw_context=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if raw_context is not None:
        headers["Content-Type"] = "application/json"
        body = '{"input": %s, "session_id": "t", "input_context": %s}' % (
            json.dumps(json.dumps(request or _BASE_REQUEST)),
            raw_context,
        )
        return client.post("/invoke", content=body.encode(), headers=headers)
    payload = {"input": json.dumps(request or _BASE_REQUEST), "session_id": "t"}
    if context is not None:
        payload["input_context"] = context
    return client.post("/invoke", json=payload, headers=headers)


class TestCallerAuthentication:
    """The deployed entry point must be reachable by the caller it declares.

    The entry node requires VERIFIED_EXTERNAL and nothing else establishes a
    trust level in a standalone deployment, so without the Bearer boundary
    every request would arrive ANONYMOUS and be denied — the agent would answer
    an error to every single call while every unit test stayed green.
    """

    def test_request_without_a_token_is_refused(self, client):
        response = _post(client, token=None)
        assert response.status_code == 401
        # The body must not say whether the token was absent, malformed or wrong.
        assert "absent" not in response.text.lower()

    def test_request_with_a_wrong_token_is_refused(self, client):
        assert _post(client, token="not-the-token").status_code == 401

    def test_authenticated_request_produces_a_document(self, client):
        response = _post(client)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success", body
        assert isinstance(body["output"], str) and body["output"].strip()
        assert "FSA Model Validation Document" in body["output"]

    def test_the_gate_ran_at_the_output_boundary(self, client):
        """post_process must appear in the trace of a successful invocation."""
        body = _post(client).json()
        assert "PostProcessNode" in body["node_history"], body["node_history"]


class TestOutputDependsOnCallerInput:
    """The published path computes from caller data — it is not a fixed answer."""

    def test_the_requested_window_reaches_the_document(self, client):
        request = dict(_BASE_REQUEST, validation_date_start="2023-03-01", validation_date_end="2024-02-01")
        output = _post(client, request=request).json()["output"]
        assert "**Validation Period:** 2023-03-01/2024-02-01" in output
        assert "**Validation Date:** 2024-02-01" in output

    def test_two_different_series_produce_two_different_reports(self, client):
        strong = [dict(r, portfolio_return=0.05) for r in _RECORDS]
        first = _post(client, context={"backtest_records": _RECORDS}).json()["output"]
        second = _post(client, context={"backtest_records": strong}).json()["output"]
        assert first != second, "the report did not depend on the record series"
        # And specifically: the drawdown differs, because one series has one.
        assert "| Max Drawdown (%) | 3.0 |" in first
        assert "| Max Drawdown (%) | 0.0 |" in second

    def test_metrics_the_document_renders_are_not_all_n_a(self, client):
        """Every performance row reads a key the metrics node actually writes.

        Two rows used to read shorter key names than the metrics node emits, so
        they rendered N/A on every run whatever had been computed.
        """
        output = _post(client, context={"backtest_records": _RECORDS}).json()["output"]
        performance = output.split("## 2. Performance Metrics")[1].split("---")[0]
        assert "N/A" not in performance, performance
        assert "None" not in performance, performance

    def test_the_record_count_and_source_are_reported(self, client):
        output = _post(client, context={"backtest_records": _RECORDS}).json()["output"]
        assert "**Records Analysed:** 4" in output
        assert "**Record Source:** caller" in output

    def test_absent_records_degrade_to_the_reference_series(self, client):
        output = _post(client).json()["output"]
        assert "**Record Source:** reference" in output


class TestCallerNumericsAreFiniteAndBounded:
    """Every caller-controlled number is parsed fail-CLOSED.

    NaN and +-Infinity are the dangerous cases: both survive raw JSON intact and
    every comparison against NaN is False, so an unvalidated NaN return would
    make each regulatory threshold read as "not violated" — a silent pass on the
    exact decision this agent exists to make.
    """

    @pytest.mark.parametrize(
        "raw_context",
        [
            '{"backtest_records":[{"portfolio_return": NaN}]}',
            '{"backtest_records":[{"portfolio_return": Infinity}]}',
            '{"backtest_records":[{"portfolio_return": -Infinity}]}',
            '{"backtest_records":[{"portfolio_return": "0.01"}]}',
            '{"backtest_records":[{"portfolio_return": true}]}',
            '{"backtest_records":[{"portfolio_return": 1000000000}]}',
            '{"backtest_records":[{"portfolio_return": -1000000000}]}',
            '{"backtest_records":[{"benchmark_return": NaN, "portfolio_return": 0.01}]}',
            '{"backtest_records":[{"cumulative_return": Infinity, "portfolio_return": 0.01}]}',
            '{"oos_months": NaN}',
            '{"oos_months": Infinity}',
            '{"oos_months": 99999}',
            '{"oos_months": "6"}',
        ],
    )
    def test_non_finite_or_mistyped_numbers_are_refused(self, client, raw_context):
        body = _post(client, raw_context=raw_context).json()
        assert body["status"] == "error", body
        _out = body.get("output") or ""
        # The refusal names the rule that stopped it: a caller handed nothing cannot tell a
        # rejected request from a hung one. What must stay absent is the ANSWER this agent
        # would have produced had the input been usable.
        assert _out.startswith("Request could not be completed."), body
        assert "FSA Model Validation Document" not in _out, body

    @pytest.mark.parametrize(
        "raw_context",
        [
            '{"backtest_records":[{"portfolio_return":0.01,"date":"not-a-date"}]}',
            '{"backtest_records":[{"portfolio_return":0.01,"date":"2024-13-01"}]}',
            '{"backtest_records":[{"benchmark_return":0.01}]}',
            '{"backtest_records":{"portfolio_return":0.01}}',
            '{"backtest_records":[[0.01]]}',
            '{"registration_date":"2025-13-40"}',
            '{"registration_date":12345}',
        ],
    )
    def test_malformed_structures_are_refused(self, client, raw_context):
        body = _post(client, raw_context=raw_context).json()
        assert body["status"] == "error", body
        _out = body.get("output") or ""
        # The refusal names the rule that stopped it: a caller handed nothing cannot tell a
        # rejected request from a hung one. What must stay absent is the ANSWER this agent
        # would have produced had the input been usable.
        assert _out.startswith("Request could not be completed."), body
        assert "FSA Model Validation Document" not in _out, body

    def test_the_record_cap_is_enforced_at_its_stated_boundary(self, client):
        from src.nodes.pre_process_node import _MAX_BACKTEST_RECORDS

        record = '{"portfolio_return":0.01}'
        at_cap = "[" + ",".join([record] * _MAX_BACKTEST_RECORDS) + "]"
        over_cap = "[" + ",".join([record] * (_MAX_BACKTEST_RECORDS + 1)) + "]"

        accepted = _post(client, raw_context='{"backtest_records":%s}' % at_cap).json()
        assert accepted["status"] == "success", accepted

        refused = _post(client, raw_context='{"backtest_records":%s}' % over_cap).json()
        assert refused["status"] == "error"
        _out = refused.get("output") or ""
        # The refusal names the rule that stopped it: a caller handed nothing cannot tell a
        # rejected request from a hung one. What must stay absent is the ANSWER this agent
        # would have produced had the input been usable.
        assert _out.startswith("Request could not be completed.")
        assert "FSA Model Validation Document" not in _out

    def test_a_rejection_never_echoes_the_rejected_value(self, client):
        """The refusal names the field; the value is caller-controlled text."""
        secret_ish = "zzz-do-not-echo-this-zzz"
        body = _post(
            client,
            request=dict(_BASE_REQUEST, asset_class=secret_ish),
        ).json()
        assert body["status"] == "error"
        assert secret_ish not in json.dumps(body)


class TestStructuredParameterCredentialScreen:
    """A credential-shaped value in the structured parameters is refused readably.

    The framework's mandatory output gate scans every value of every node
    result, and the backbone's first node copies the structured parameters
    verbatim into its own result — so such a value makes the FIRST node fail
    before any template code runs, and the caller gets an error naming nothing.
    Declaring an inert contract is not immunity: validators IGNORE undeclared
    keys, and ignoring is not stripping.
    """

    @pytest.mark.parametrize(
        "context",
        [
            {"note": "Bearer abcdef1234567890abcdef"},
            {"undeclared_blob": "AKIA1234567890ABCDEF"},
            {"nested": {"deep": {"key": "sk_live_" + "abcdefghij0123456789"}}},
            {"dsn": _CONNECTION_URI},
            {"list_field": ["fine", "sk-abcdefghijklmnopqrstuvwxyz"]},
        ],
    )
    def test_credential_shaped_values_are_refused_by_field(self, client, context):
        response = _post(client, context=context)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "input_context" in detail
        # The matched value is never echoed back.
        for value in context.values():
            assert json.dumps(value) not in detail

    def test_the_refusal_set_equals_the_framework_detector(self, client):
        """Per-field screening composes exactly to scanning the whole mapping.

        The detector on a mapping is the union over its values, so scanning
        field by field neither widens nor narrows the block set. Pinning the
        identity is what stops the screen drifting away from the gate that
        would otherwise fail the request opaquely.
        """
        from framework.security.credential_detector import detect_credentials_in_value

        contexts = [
            {"note": "Bearer abcdef1234567890abcdef"},
            {"undeclared_blob": "AKIA1234567890ABCDEF"},
            {"backtest_records": _RECORDS},
            {"registration_date": "2025-01-01"},
            {},
        ]
        for context in contexts:
            refused = _post(client, context=context).status_code == 400
            assert refused == bool(detect_credentials_in_value(context)), context

    def test_an_unsafe_field_name_is_reported_by_position(self, client):
        response = _post(
            client,
            context={"a name with spaces": "AKIA1234567890ABCDEF"},
        )
        assert response.status_code == 400
        assert "input_context field #1" in response.json()["detail"]

    def test_ordinary_domain_text_on_the_same_channel_still_passes(self, client):
        response = _post(
            client,
            context={"backtest_records": _RECORDS, "registration_date": "2025-01-01"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_oversized_structured_parameters_are_refused(self, client):
        response = _post(client, context={"padding": "x" * 300_000})
        assert response.status_code == 413


class TestDeclaredConfigurationIsLive:
    """A value written in config/config.yaml must reach the node that reads it.

    The entry point used to construct the graph with no configuration at all,
    so every declared runtime value was silently ignored and the code ran on
    its built-in defaults. That failure mode is invisible unless a declared
    value is observed changing the answer end to end.
    """

    def _invoke_with_threshold(self, monkeypatch, threshold):
        from fastapi.testclient import TestClient

        import src.api.server as server
        from src.graph.graph import Graph, _domain_settings

        config = dict(server.runtime_config())
        config["fsa_thresholds"] = dict(config.get("fsa_thresholds", {}))
        config["fsa_thresholds"]["fsa_chk01_min_sharpe"] = threshold
        agent = Graph(config=config)
        agent.compile()
        monkeypatch.setattr(server, "agent", agent)
        monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)
        client = TestClient(server.app)
        body = _post(
            client,
            context={
                "backtest_records": _RECORDS,
                "oos_months": 8,
                "registration_date": "2025-01-01",
            },
        ).json()
        assert _domain_settings is not None
        return body["output"] or ""

    def test_a_declared_threshold_changes_the_verdict(self, monkeypatch):
        lenient = self._invoke_with_threshold(monkeypatch, 0.5)
        strict = self._invoke_with_threshold(monkeypatch, 90.0)
        assert "| FSA-CHK-01 | Sharpe Ratio | PASS |" in lenient
        assert "| FSA-CHK-01 | Sharpe Ratio | FAIL |" in strict

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "90.0", True, None, -1e6])
    def test_a_malformed_declared_threshold_falls_back_to_the_default(self, monkeypatch, bad):
        """A bad declaration must not disable the check it feeds.

        NaN is the case that matters: it compares False against everything, so
        an unvalidated NaN threshold would turn the check into a silent pass.
        """
        output = self._invoke_with_threshold(monkeypatch, bad)
        assert "| FSA-CHK-01 | Sharpe Ratio | PASS |" in output

    def test_the_entry_point_loads_the_declared_configuration(self):
        from src.graph.graph import runtime_config

        config = runtime_config()
        assert config.get("max_retry") == 3
        assert config.get("timeout_s") == 30
        assert "fsa_thresholds" in config

    def test_the_deployed_agent_was_built_with_that_configuration(self):
        """The entry point must PASS the file it loads to the graph.

        Constructing the graph with no configuration reads as harmless — the
        code keeps working on its defaults — so nothing fails until a declared
        value is expected to matter. This asserts the wiring itself.
        """
        import src.api.server as server

        assert server.agent.config.get("max_retry") == 3
        assert server.agent.config.get("timeout_s") == 30
        assert server.agent.config.get("fsa_thresholds"), (
            "the deployed agent holds no declared thresholds — the entry point "
            "built the graph without its configuration"
        )

    def test_the_declared_threshold_reaches_the_node_that_reads_it(self):
        """Follow one declared value all the way to the node instance.

        Loading the file, forwarding it to the inner graph and handing it to
        the node are three separate steps; a break in any one of them leaves
        the declaration inert while every other test stays green.
        """
        import src.api.server as server
        from src.graph.graph import runtime_config

        declared = runtime_config()["fsa_thresholds"]["fsa_chk01_min_sharpe"]

        main_node = server.agent._nodes["main"]
        inner = main_node.get_subgraph()
        inner.register_nodes()
        metrics_node = inner._nodes["validation_metrics"]

        assert metrics_node._thresholds["fsa_chk01_min_sharpe"] == declared
