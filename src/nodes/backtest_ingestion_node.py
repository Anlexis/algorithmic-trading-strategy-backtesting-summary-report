"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from api/ or from another agent
#
# FIN-C2-051 — BacktestIngestionNode
# Inner domain graph node 1: assemble the backtest record series, model
# metadata and benchmark series for the requested model and window.
#
# Input:  state["validated_input"] — dict with model_id, the validation window
#         (start_date / end_date), asset_class, and the validated
#         backtest_records series when the caller supplied one
# Output: state["backtest_results"], state["model_metadata"],
#         state["benchmark_data"], state["result"]
#
# Record source, in order: the caller's validated series when present, then a
# constructor-injected data-source adapter, then the built-in reference series.
# Missing data degrades to the reference series rather than raising.

import logging
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail. `emit_trace_event` is the free-function audit API and ships in
# the framework distribution; FunctionNode exposes no instance method of that
# name, so the method form would raise AttributeError. Use the free function.
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Model identifiers this template is authorised to validate.
_AUTHORIZED_MODEL_IDS: List[str] = [
    "MODEL-ALPHA",
    "MODEL-BETA",
    "MODEL-GAMMA",
    "MODEL-DELTA",
    "MODEL-EPSILON",
]
_MAX_BACKTEST_WINDOW_MONTHS = 60


def _months_between(start_date: str, end_date: str) -> int:
    """Return approximate number of months between two YYYY-MM-DD date strings."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return max(0, (end.year - start.year) * 12 + (end.month - start.month))
    except ValueError:
        return 0


def _load_reference_data(
    model_id: str,
    start_date: str,
    end_date: str,
    asset_class: str,
) -> Dict[str, Any]:
    """Return the built-in reference series for the given model and window.

    Used when the caller supplied no record series and no data-source adapter is
    configured. It is a fixed two-point series: it exercises every downstream
    computation without standing in for real data, and the generated document
    says which source it came from.
    """
    backtest_results: List[Dict[str, Any]] = [
        {
            "date": start_date,
            "portfolio_return": 0.023,
            "benchmark_return": 0.018,
            "alpha": 0.005,
            "sharpe_ratio": 1.42,
        },
        {
            "date": end_date,
            "portfolio_return": 0.031,
            "benchmark_return": 0.025,
            "alpha": 0.006,
            "sharpe_ratio": 1.55,
        },
    ]

    model_metadata: Dict[str, Any] = {
        "model_name": model_id,
        "version": "1.0.0",
        "strategy_type": "momentum",
        "asset_class": asset_class,
        "backtest_period_start": start_date,
        "backtest_period_end": end_date,
    }

    benchmark_data: List[Dict[str, Any]] = [
        {
            "date": start_date,
            "benchmark_return": 0.018,
            "index": "TOPIX",
        },
        {
            "date": end_date,
            "benchmark_return": 0.025,
            "index": "TOPIX",
        },
    ]

    return {
        "backtest_results": backtest_results,
        "model_metadata": model_metadata,
        "benchmark_data": benchmark_data,
    }


def _apply_input_gate(state: Dict[str, Any]) -> Optional[str]:
    """Domain input gate — a module-level helper, enforced inline from execute().

    It is deliberately NOT a `_extra_security_gate_input` override: the
    framework auto-wraps that hook into the graph chain, and on the clean path
    the wrapped hook returns None, so the next node receives ``state=None`` and
    the invocation raises AttributeError. Calling the check inline from
    execute() keeps every rule running without that failure mode.

    Returns a violation reason, or None when the input is acceptable.

    Checks:
        - model_id must be in the authorised allowlist
        - the requested window must not exceed the maximum allowed months
    """
    validated_input: Dict[str, Any] = state.get("validated_input") or {}
    model_id: str = str(validated_input.get("model_id", "")).strip()
    start_date: str = str(validated_input.get("start_date", "")).strip()
    end_date: str = str(validated_input.get("end_date", "")).strip()

    if model_id and model_id not in _AUTHORIZED_MODEL_IDS:
        return (
            "BacktestIngestionNode: 'model_id' is not in the authorised allowlist. "
            f"Authorised identifiers: {_AUTHORIZED_MODEL_IDS}"
        )

    if start_date and end_date:
        window_months = _months_between(start_date, end_date)
        if window_months > _MAX_BACKTEST_WINDOW_MONTHS:
            return (
                f"BacktestIngestionNode: the requested window ({window_months} months) "
                f"exceeds the maximum allowed ({_MAX_BACKTEST_WINDOW_MONTHS} months)."
            )

    return None


class BacktestIngestionNode(FunctionNode):
    """Assemble backtest records, model metadata and benchmark data for the model/window.

    Inner domain graph node 1 (registered as "backtest_ingestion" in
    DomainWorkflowGraph). Accepts the caller's validated record series, an
    injected data-source adapter, or the built-in reference series — in that
    order — and never raises on missing data.

    Input state keys:
        validated_input: dict with model_id, start_date, end_date, asset_class
                         and (optionally) backtest_records / registration_date

    Output state keys (partial dict):
        backtest_results: list of period records
                          [{date, portfolio_return, benchmark_return, ...}]
        model_metadata:   {model_name, version, strategy_type, asset_class,
                           backtest_period_start, backtest_period_end,
                           data_source, registration_date}
        benchmark_data:   list of benchmark return records for comparison
        result:           ingest summary string
        status:           AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:        (on error) list of error messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, data_source: Optional[Any] = None) -> None:
        """Initialise the node with an optional data source adapter.

        Args:
            data_source: External data source adapter (duck-typed; must expose
                ``load(model_id, start_date, end_date, asset_class) -> dict``).
                When None, and the caller supplied no record series, the
                built-in reference series is used.
        """
        super().__init__()
        self._data_source = data_source

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        validated_input: Optional[Dict[str, Any]] = state.get("validated_input")

        if not validated_input or not isinstance(validated_input, dict):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "BacktestIngestionNode: 'validated_input' is missing or not an "
                    "object — the input validation step did not run"
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (
                    "BacktestIngestionNode: 'validated_input' is missing or not an object — the input validation step did not run"
                ),
            }

        # Domain input gate — see _apply_input_gate for why it runs inline.
        violation = _apply_input_gate(state)
        if violation:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [violation],
                # NO P1 HERE. The gate's message lists the whole authorised model-id allowlist
                # ("Authorised identifiers: ..."). In error_log that never leaves the graph;
                # surfaced to the caller it hands an internal registry to anyone who guesses one
                # wrong id. This refusal stays silent.
            }

        model_id: str = str(validated_input.get("model_id", "")).strip()
        start_date: str = str(validated_input.get("start_date") or "").strip()
        end_date: str = str(validated_input.get("end_date") or "").strip()
        asset_class: str = str(validated_input.get("asset_class") or "equity").strip()

        if not model_id:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["BacktestIngestionNode: 'model_id' is missing from validated_input"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("BacktestIngestionNode: 'model_id' is missing from validated_input"),
            }
        if not start_date or not end_date:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["BacktestIngestionNode: 'start_date' and 'end_date' are required " "in validated_input"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("BacktestIngestionNode: 'start_date' and 'end_date' are required in validated_input"),
            }

        caller_records: List[Dict[str, Any]] = [
            record for record in (validated_input.get("backtest_records") or []) if isinstance(record, dict)
        ]

        logger.info(
            "BacktestIngestionNode: assembling data — model_id=%s start=%s end=%s " "asset_class=%s caller_records=%d",
            model_id,
            start_date,
            end_date,
            asset_class,
            len(caller_records),
        )

        try:
            if caller_records:
                # The caller supplied a validated series: every number in it has
                # already passed the finite + bounded parser, so it is used as
                # given and the reference series is not consulted at all.
                data_source_name = "caller"
                backtest_results: List[Dict[str, Any]] = caller_records
                benchmark_data: List[Dict[str, Any]] = [
                    {"date": record.get("date", ""), "benchmark_return": record["benchmark_return"]}
                    for record in caller_records
                    if "benchmark_return" in record
                ]
                model_metadata: Dict[str, Any] = {
                    "model_name": model_id,
                    "version": str(validated_input.get("model_version") or "1.0.0"),
                    "strategy_type": "caller_supplied",
                    "asset_class": asset_class,
                    "backtest_period_start": start_date,
                    "backtest_period_end": end_date,
                }
            elif self._data_source is not None:
                data_source_name = "adapter"
                raw = self._data_source.load(model_id, start_date, end_date, asset_class)
                backtest_results = raw.get("backtest_results", [])
                model_metadata = raw.get("model_metadata", {})
                benchmark_data = raw.get("benchmark_data", [])
            else:
                data_source_name = "reference"
                data = _load_reference_data(model_id, start_date, end_date, asset_class)
                backtest_results = data.get("backtest_results", [])
                model_metadata = data.get("model_metadata", {})
                benchmark_data = data.get("benchmark_data", [])
        except Exception as exc:  # noqa: BLE001
            logger.error("BacktestIngestionNode: data load failed: %s", exc)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["BacktestIngestionNode: the data source could not be read"],
                "backtest_results": [],
                "model_metadata": {},
                "benchmark_data": [],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("BacktestIngestionNode: the data source could not be read"),
            }

        if isinstance(model_metadata, dict):
            model_metadata = dict(model_metadata)
            model_metadata.setdefault("model_id", model_id)
            model_metadata["data_source"] = data_source_name
            registration_date = validated_input.get("registration_date")
            if registration_date:
                model_metadata["registration_date"] = registration_date

        period_months = _months_between(start_date, end_date)
        record_count = len(backtest_results)

        logger.info(
            "BacktestIngestionNode: ingested model_id=%s records=%d benchmark=%d " "period_months=%d source=%s",
            model_id,
            record_count,
            len(benchmark_data),
            period_months,
            data_source_name,
        )

        # Audit trail: record the ingest side-effect.
        emit_trace_event(
            "backtest_ingestion_complete",
            {
                "record_count": record_count,
                "model_id": model_id,
                "period_months": period_months,
                "data_source": data_source_name,
            },
            state,
        )

        result_summary = (
            f"Loaded {record_count} backtest records for {model_id} "
            f"({start_date} – {end_date}, {asset_class}; source: {data_source_name})"
        )

        return {
            "backtest_results": backtest_results,
            "model_metadata": model_metadata,
            "benchmark_data": benchmark_data,
            "result": result_summary,
            "status": AgentStatus.SUCCESS.value,
        }
