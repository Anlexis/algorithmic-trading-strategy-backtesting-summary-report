# FIN-C2-051 — Unit Tests: BacktestIngestionNode (TC-11 + data loading BL)
#
# TC-11: emit_trace_event fires on the success path (audit trail)
# Business logic:
#   - Built-in stub data source populates all 3 output fields
#   - Constructor-injectable data_source adapter is exercised (data_source=...)
#   - Missing / malformed validated_input returns ERROR (no exception)

from unittest.mock import MagicMock


def _status_ok(result) -> bool:
    from framework.schemas.agent_status import AgentStatus

    return result.get("status") in (AgentStatus.SUCCESS, AgentStatus.SUCCESS.value)


def _status_err(result) -> bool:
    from framework.schemas.agent_status import AgentStatus

    return result.get("status") in (AgentStatus.ERROR, AgentStatus.ERROR.value)


def _validated_input(model_id="MODEL-ALPHA", start_date="2024-01-01", end_date="2025-12-01", asset_class="equity"):
    return {
        "model_id": model_id,
        "start_date": start_date,
        "end_date": end_date,
        "asset_class": asset_class,
    }


class TestBacktestIngestionStubData:
    """BacktestIngestionNode: built-in stub source populates all output fields."""

    def test_stub_data_populates_all_fields(self):
        """Default stub source populates backtest_results, model_metadata, benchmark_data."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        node = BacktestIngestionNode()
        result = node.execute({"validated_input": _validated_input()})

        assert _status_ok(result), f"Expected SUCCESS, error_log: {result.get('error_log')}"
        assert isinstance(result.get("backtest_results"), list), "backtest_results must be a list"
        assert len(result.get("backtest_results", [])) > 0, "stub must return >= 1 record"
        assert isinstance(result.get("model_metadata"), dict), "model_metadata must be a dict"
        assert isinstance(result.get("benchmark_data"), list), "benchmark_data must be a list"
        assert result.get("result"), "result summary string must be non-empty"

    def test_model_metadata_carries_model_id(self):
        """model_metadata.model_name must reflect the requested model_id."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        node = BacktestIngestionNode()
        result = node.execute({"validated_input": _validated_input(model_id="MODEL-BETA")})

        assert _status_ok(result)
        meta = result.get("model_metadata", {})
        assert (
            meta.get("model_name") == "MODEL-BETA"
        ), f"model_metadata.model_name must be MODEL-BETA, got {meta.get('model_name')}"
        assert meta.get("asset_class") == "equity"


class TestBacktestIngestionInjectableDataSource:
    """data_source=... constructor injection: the adapter's load() is exercised."""

    def test_injected_data_source_load_is_called(self):
        """An injected data_source.load(...) must be called with the right args."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        fake_source = MagicMock()
        fake_source.load.return_value = {
            "backtest_results": [{"date": "2024-01-01", "portfolio_return": 0.01}],
            "model_metadata": {"model_name": "MODEL-GAMMA", "asset_class": "fx"},
            "benchmark_data": [{"date": "2024-01-01", "benchmark_return": 0.005}],
        }

        node = BacktestIngestionNode(data_source=fake_source)
        vi = _validated_input(model_id="MODEL-GAMMA", asset_class="fx")
        result = node.execute({"validated_input": vi})

        assert _status_ok(result), f"Expected SUCCESS, got {result}"
        fake_source.load.assert_called_once_with("MODEL-GAMMA", vi["start_date"], vi["end_date"], "fx")
        assert result.get("backtest_results") == [{"date": "2024-01-01", "portfolio_return": 0.01}]

    def test_data_source_load_exception_returns_error(self):
        """If the injected data_source.load raises, the node returns ERROR (no crash)."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        fake_source = MagicMock()
        fake_source.load.side_effect = RuntimeError("connection refused")

        node = BacktestIngestionNode(data_source=fake_source)
        result = node.execute({"validated_input": _validated_input()})

        assert _status_err(result), "data load failure must surface as ERROR, not raise"
        assert result.get("error_log"), "error_log must describe the failure"


class TestBacktestIngestionErrorPaths:
    """Missing / malformed validated_input returns ERROR gracefully."""

    def test_missing_validated_input_returns_error(self):
        """Absent validated_input must return ERROR (PreProcess gate did not run)."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        node = BacktestIngestionNode()
        result = node.execute({})

        assert _status_err(result)

    def test_non_dict_validated_input_returns_error(self):
        """validated_input that is not a dict must return ERROR."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        node = BacktestIngestionNode()
        result = node.execute({"validated_input": "not a dict"})

        assert _status_err(result)

    def test_missing_model_id_returns_error(self):
        """validated_input without model_id must return ERROR."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        node = BacktestIngestionNode()
        vi = _validated_input()
        vi.pop("model_id")
        result = node.execute({"validated_input": vi})

        assert _status_err(result)

    def test_missing_date_range_returns_error(self):
        """validated_input without start_date/end_date must return ERROR."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode

        node = BacktestIngestionNode()
        vi = _validated_input()
        vi.pop("start_date")
        vi.pop("end_date")
        result = node.execute({"validated_input": vi})

        assert _status_err(result)


class TestTC11EmitTraceEvent:
    """TC-11: emit_trace_event fires on the success path with the node's event."""

    def test_emit_trace_event_called_on_success(self, monkeypatch):
        """Patch emit_trace_event in the node module; assert it fires once on success."""
        import src.nodes.backtest_ingestion_node as mod

        calls = []
        monkeypatch.setattr(mod, "emit_trace_event", lambda event, payload, st: calls.append(event))

        node = mod.BacktestIngestionNode()
        result = node.execute({"validated_input": _validated_input()})

        assert _status_ok(result)
        assert (
            "backtest_ingestion_complete" in calls
        ), f"TC-11: expected audit event 'backtest_ingestion_complete', got {calls}"

    def test_emit_trace_event_fires_exactly_once(self, monkeypatch):
        """The audit event must fire exactly once per successful invocation."""
        import src.nodes.backtest_ingestion_node as mod

        calls = []
        monkeypatch.setattr(mod, "emit_trace_event", lambda event, payload, st: calls.append(event))

        node = mod.BacktestIngestionNode()
        node.execute({"validated_input": _validated_input()})

        assert len(calls) == 1, f"TC-11: emit_trace_event must fire once, fired {len(calls)}×"
