# FIN-C2-051 — Proof-of-Boundary Tests (PB-1 .. PB-6)
#
# PB-1: BaseNode → EventEmitter: emit_trace_event() fires on every node's
#       success path (all 5 nodes: PreProcess, BacktestIngestion,
#       ValidationMetrics, DocumentGeneration, PostProcess) — the audit trail.
# PB-2: State serialization: post-invoke State output fields are JSON-serializable
#       primitives (no Pydantic / dataclass — msgpack safe).
# PB-3: External service boundary: BacktestIngestionNode's injectable data
#       source interface is exercised (reference series + injected adapter).
# PB-4: Import isolation: AST scan asserting no `agenticstar` imports under src/.
# PB-5: Checkpoint safety: no JWT / Pydantic objects in serialized State output.
# PB-6: Invoke execution order: FunctionNode subclasses must not override the
#       @final security-gate methods (trust gate → node_start → input gate →
#       execute → output gate → node_complete order).
#
# emit_trace_event is the free-function audit API, imported into each node
# module via `from shared.utils.audit_logger import emit_trace_event`. That
# package ships in the AgentCore SDK wheel ONLY (not in repo source), so a bare
# local checkout cannot import the node modules — those classes skip locally and
# are verified in CI, exactly like the other boundary tests. When the wheel IS
# present (CI), each node module exposes `emit_trace_event` in its own namespace,
# so we patch THAT name with a spy and assert the audit event fired.

import ast
import importlib
import json
import os
import re

import pytest


# ---------------------------------------------------------------------------
# Shared success-path drivers, keyed to each node's real execute() contract.
# ---------------------------------------------------------------------------


def _pre_process_state():
    return {
        "user_input": json.dumps(
            {
                "model_id": "MODEL-ALPHA",
                "asset_class": "equity",
                "backtest_data": "path/to/backtest.json",
            }
        )
    }


def _backtest_ingestion_state():
    return {
        "validated_input": {
            "model_id": "MODEL-ALPHA",
            "start_date": "2024-01-01",
            "end_date": "2025-12-01",
            "asset_class": "equity",
        }
    }


def _validation_metrics_state():
    return {
        "backtest_results": {
            "returns": [0.01, 0.02, 0.015, 0.005, 0.012, 0.008, 0.018, 0.011],
            "cumulative": [1.0, 1.2, 0.9, 1.1],
            "win_count": 5,
            "total_count": 8,
            "backtest_months": 36,
            "oos_months": 12,
            "registration_date": "2026-01-01",
            "backtest_start_date": "2020-01-01",
        },
        "benchmark_data": {"returns": [0.008] * 8},
        "model_metadata": {"registration_date": "2026-01-01"},
    }


def _document_generation_state():
    return {
        "performance_metrics": {"sharpe_ratio": 1.42, "max_drawdown": 0.12},
        "risk_metrics": {"VaR_95": 0.03, "volatility": 0.14},
        "fsa_checks": [
            {"check_id": "FSA-CHK-01", "check_name": "Sharpe Ratio", "result": "PASS", "detail": "Sharpe=1.42"},
        ],
        "model_metadata": {
            "model_id": "MODEL-ALPHA",
            "version": "1.0.0",
            "asset_class": "equity",
            "strategy_type": "momentum",
        },
        "validated_input": {"period": "2024-01/2025-12", "validation_date": "2026-01-01"},
    }


def _post_process_state():
    return {
        "result": "# FSA Validation Document\n\nClean report body, all checks passed.",
        "document_metadata": {
            "model_id": "MODEL-ALPHA",
            "generated_at": "2026-01-01",
            "validation_date": "2026-01-01",
            "fsa_check_pass_count": 5,
            "fsa_check_fail_count": 0,
        },
        "model_id": "MODEL-ALPHA",
    }


# (node module path, class name, expected audit event name, success-path state)
_S4_NODE_CASES = [
    ("src.nodes.pre_process_node", "PreProcessNode", "pre_process_validated", _pre_process_state()),
    (
        "src.nodes.backtest_ingestion_node",
        "BacktestIngestionNode",
        "backtest_ingestion_complete",
        _backtest_ingestion_state(),
    ),
    (
        "src.nodes.validation_metrics_node",
        "ValidationMetricsNode",
        "validation_metrics_calculated",
        _validation_metrics_state(),
    ),
    (
        "src.nodes.document_generation_node",
        "DocumentGenerationNode",
        "document_generation_complete",
        _document_generation_state(),
    ),
    ("src.nodes.post_process_node", "PostProcessNode", "post_process_complete", _post_process_state()),
]


# ---------------------------------------------------------------------------
# PB-1: BaseNode → EventEmitter — the audit trail (emit_trace_event)
# ---------------------------------------------------------------------------


class TestPB1AuditTrailEmitTraceEvent:
    """PB-1: every node emits its audit event exactly once on the success path."""

    @pytest.mark.parametrize(
        "module_path, class_name, event_name, state",
        _S4_NODE_CASES,
        ids=[c[1] for c in _S4_NODE_CASES],
    )
    def test_node_emits_s4_audit_event(self, monkeypatch, module_path, class_name, event_name, state):
        """Patch emit_trace_event in the node module and assert it fires once
        with the node's domain event name on a successful execute()."""
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip(
                "shared.utils.audit_logger / framework not installed "
                "(SDK wheel-only) — PB-1 audit path is verified in CI"
            )

        calls = []

        def _spy(event, payload, st):
            calls.append((event, payload, st))

        # The name lives in the node module's namespace (imported by name).
        monkeypatch.setattr(module, "emit_trace_event", _spy)

        node = getattr(module, class_name)()
        node.execute(dict(state))

        assert calls, f"PB-1: {class_name}.execute() did not emit any audit event"
        emitted_names = [c[0] for c in calls]
        assert (
            event_name in emitted_names
        ), f"PB-1: {class_name} expected audit event {event_name!r}, got {emitted_names!r}"
        # Fires exactly once per node on the success path (no duplicate events).
        assert emitted_names.count(event_name) == 1, (
            f"PB-1: {class_name} must emit {event_name!r} exactly once, " f"got {emitted_names!r}"
        )
        # Event name must be a non-empty string and payload a dict.
        for ev, payload, _st in calls:
            assert isinstance(ev, str) and ev, "PB-1: event name must be non-empty str"
            assert isinstance(payload, dict), "PB-1: event payload must be a dict"

    def test_pb1_covers_all_five_nodes(self):
        """PB-1 must exercise all 5 nodes (3 domain + 2 outer backbone)."""
        names = {c[1] for c in _S4_NODE_CASES}
        assert names == {
            "PreProcessNode",
            "BacktestIngestionNode",
            "ValidationMetricsNode",
            "DocumentGenerationNode",
            "PostProcessNode",
        }, f"PB-1 must cover all 5 nodes, got {names}"


# ---------------------------------------------------------------------------
# PB-2: State serialization — output fields are JSON-serializable primitives
# ---------------------------------------------------------------------------


class TestPB2StateSerializationSafety:
    """PB-2: node output fields must be JSON-serializable (msgpack-compatible)."""

    def _is_json_serializable(self, obj) -> bool:
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _run(self, module_path, class_name, state):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip("framework not installed (SDK wheel-only) — verified in CI")
        node = getattr(module, class_name)()
        return node.execute(dict(state))

    def test_backtest_ingestion_output_json_serializable(self):
        """BacktestIngestionNode output fields must be JSON-serializable."""
        result = self._run(
            "src.nodes.backtest_ingestion_node",
            "BacktestIngestionNode",
            _backtest_ingestion_state(),
        )
        for field in ("backtest_results", "model_metadata", "benchmark_data", "result"):
            value = result.get(field)
            assert self._is_json_serializable(
                value
            ), f"PB-2: BacktestIngestionNode field '{field}' not JSON-serializable: {type(value)}"

    def test_validation_metrics_output_json_serializable(self):
        """ValidationMetricsNode output fields must be JSON-serializable."""
        result = self._run(
            "src.nodes.validation_metrics_node",
            "ValidationMetricsNode",
            _validation_metrics_state(),
        )
        for field in ("performance_metrics", "risk_metrics", "fsa_checks"):
            value = result.get(field)
            assert self._is_json_serializable(
                value
            ), f"PB-2: ValidationMetricsNode field '{field}' not JSON-serializable: {type(value)}"

    def test_document_generation_output_json_serializable(self):
        """DocumentGenerationNode output fields must be JSON-serializable."""
        result = self._run(
            "src.nodes.document_generation_node",
            "DocumentGenerationNode",
            _document_generation_state(),
        )
        for field in ("validation_document", "llm_response", "result", "document_metadata"):
            value = result.get(field)
            assert self._is_json_serializable(
                value
            ), f"PB-2: DocumentGenerationNode field '{field}' not JSON-serializable: {type(value)}"


# ---------------------------------------------------------------------------
# PB-3: External service boundary: BacktestIngestionNode data-source interface
# ---------------------------------------------------------------------------


class TestPB3DataSourceConnectionInterface:
    """PB-3: BacktestIngestionNode's external data source connection is exercised."""

    def test_stub_data_source_returns_records(self):
        """Default (stub) source retrieves non-empty backtest records."""
        try:
            from src.nodes.backtest_ingestion_node import BacktestIngestionNode
            from framework.schemas.agent_status import AgentStatus
        except ImportError:
            pytest.skip("framework not installed (SDK wheel-only) — verified in CI")

        node = BacktestIngestionNode()
        result = node.execute(_backtest_ingestion_state())

        assert result.get("status") in (
            AgentStatus.SUCCESS,
            AgentStatus.SUCCESS.value,
        ), f"PB-3: BacktestIngestionNode failed with stub source: {result.get('error_log')}"
        records = result.get("backtest_results", [])
        assert isinstance(records, list), "backtest_results must be a list"
        assert len(records) > 0, "PB-3: BacktestIngestionNode must retrieve >= 1 record from the stub source"

    def test_injected_data_source_interface_exercised(self):
        """An injected data_source adapter's load() connection interface is called."""
        try:
            from src.nodes.backtest_ingestion_node import BacktestIngestionNode
        except ImportError:
            pytest.skip("framework not installed (SDK wheel-only) — verified in CI")

        from unittest.mock import MagicMock

        fake_source = MagicMock()
        fake_source.load.return_value = {
            "backtest_results": [{"date": "2024-01-01", "portfolio_return": 0.01}],
            "model_metadata": {"model_name": "MODEL-ALPHA", "asset_class": "equity"},
            "benchmark_data": [{"date": "2024-01-01", "benchmark_return": 0.005}],
        }

        node = BacktestIngestionNode(data_source=fake_source)
        node.execute(_backtest_ingestion_state())

        # The boundary being proven: the connection interface (.load) was invoked.
        assert fake_source.load.called, "PB-3: injected data_source.load() connection interface must be exercised"


# ---------------------------------------------------------------------------
# PB-4: Import isolation — AST scan asserting no `agenticstar` imports under src/
# ---------------------------------------------------------------------------


class TestPB4ImportIsolation:
    """PB-4: no Level-0 (agenticstar) imports anywhere under src/."""

    def _src_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "..", "..", "src")

    def _scan_file(self, filepath: str) -> list:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "agenticstar" or alias.name.startswith("agenticstar."):
                        violations.append(f"{filepath}:{node.lineno} — import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "agenticstar" or mod.startswith("agenticstar."):
                    violations.append(f"{filepath}:{node.lineno} — from {mod} import ...")
        return violations

    def test_no_agenticstar_imports_under_src(self):
        """AST-scan every .py under src/ and assert no agenticstar imports."""
        src_dir = self._src_dir()
        if not os.path.exists(src_dir):
            pytest.skip("src/ directory not found")

        violations = []
        for root, _dirs, files in os.walk(src_dir):
            for name in files:
                if name.endswith(".py"):
                    violations.extend(self._scan_file(os.path.join(root, name)))

        assert violations == [], "PB-4: Level-0 import isolation violations found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# PB-5: Checkpoint safety — no Pydantic / JWT-like objects in State output
# ---------------------------------------------------------------------------


class TestPB5CheckpointSafety:
    """PB-5: serialized State output must not contain Pydantic or JWT-like objects."""

    _JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")

    def _run(self, module_path, class_name, state):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip("framework not installed (SDK wheel-only) — verified in CI")
        node = getattr(module, class_name)()
        return node.execute(dict(state))

    def test_no_pydantic_objects_in_document_output(self):
        """DocumentGenerationNode output must not contain Pydantic BaseModel instances."""
        try:
            from pydantic import BaseModel as PydanticBase
        except ImportError:
            pytest.skip("Pydantic not installed — isolation already confirmed by PB-4")

        result = self._run(
            "src.nodes.document_generation_node",
            "DocumentGenerationNode",
            _document_generation_state(),
        )

        def _check(obj, path="root"):
            if isinstance(obj, PydanticBase):
                raise AssertionError(f"PB-5: Pydantic object at '{path}': {type(obj)}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _check(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _check(item, f"{path}[{i}]")

        for field, value in result.items():
            _check(value, field)

    def test_no_jwt_like_strings_in_document_output(self):
        """DocumentGenerationNode output must not contain JWT-like strings."""
        result = self._run(
            "src.nodes.document_generation_node",
            "DocumentGenerationNode",
            _document_generation_state(),
        )

        def _check(obj, path="root"):
            if isinstance(obj, str):
                if self._JWT_RE.search(obj):
                    raise AssertionError(f"PB-5: JWT-like string at '{path}': {obj[:40]!r}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _check(item, f"{path}[{i}]")

        for field, value in result.items():
            _check(value, field)


# ---------------------------------------------------------------------------
# PB-6: Invoke execution order — @final security-gate methods not overridden
# ---------------------------------------------------------------------------
#
# The framework BaseNode.__call__() enforces the security order:
#   trust gate → node_start → _security_gate_input → execute()
#   → _security_gate_output → node_complete.
# A subclass that overrode _security_gate_input / _security_gate_output would
# break that order. Domain nodes may only extend via the _extra_* hooks.


class TestPB6InvokeExecutionOrder:
    """PB-6: FunctionNode subclasses must not override the @final gate methods."""

    _NODES = [
        ("src.nodes.pre_process_node", "PreProcessNode"),
        ("src.nodes.backtest_ingestion_node", "BacktestIngestionNode"),
        ("src.nodes.validation_metrics_node", "ValidationMetricsNode"),
        ("src.nodes.document_generation_node", "DocumentGenerationNode"),
        ("src.nodes.post_process_node", "PostProcessNode"),
    ]

    def _check_no_override(self, NodeClass, method_name):
        """Assert NodeClass does not override a @final framework gate method.

        Only the `_extra_`-prefixed extension hooks are permitted overrides.
        """
        if method_name in NodeClass.__dict__ and not method_name.startswith("_extra_"):
            pytest.fail(
                f"PB-6: {NodeClass.__name__} overrides @final method "
                f"'{method_name}' — this violates the framework security order"
            )

    @pytest.mark.parametrize("module_path, class_name", _NODES)
    def test_node_does_not_override_final_gates(self, module_path, class_name):
        """Each node must not override _security_gate_input / _security_gate_output."""
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip("framework not installed (SDK wheel-only) — verified in CI")

        NodeClass = getattr(module, class_name)
        self._check_no_override(NodeClass, "_security_gate_input")
        self._check_no_override(NodeClass, "_security_gate_output")

    @pytest.mark.parametrize("module_path, class_name", _NODES)
    def test_node_implements_execute(self, module_path, class_name):
        """Each node must implement execute() (not _invoke_impl)."""
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            pytest.skip("framework not installed (SDK wheel-only) — verified in CI")

        NodeClass = getattr(module, class_name)
        assert "execute" in NodeClass.__dict__, f"{class_name} must define execute()"
        assert "_invoke_impl" not in NodeClass.__dict__, f"{class_name} must NOT define _invoke_impl() — use execute()"
