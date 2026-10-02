"""AgentCore Platform v1.0"""

# FIN-C2-051 — PreProcessNode
# Outer backbone pre_process slot — input validation gate.
#
# Validates the invocation payload for algorithmic trading backtest validation:
# the model identifier, the asset class, the validation window, and the
# caller-supplied backtest record series.
#
# Rules:
#   - Reject missing / non-string / non-object user_input
#   - Validate model_id against an inert identifier alphabet (<=64 chars)
#   - Validate asset_class against a closed allowlist
#   - Reject a validation_date_end in the future
#   - Validate every caller-supplied number through a finite + bounded parser
#     (NaN and +-Infinity parse fine through json/float and compare False
#     against every threshold, which would fail silently OPEN)
#   - Cap the number of accepted backtest records and the inline data size
#   - Name the offending FIELD in every rejection; never echo the value
#
# Audit: emit_trace_event("pre_process_validated", {...}, state).

import json
import logging
import math
import re
from datetime import date, datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail — the free-function form; the instance-method form does not exist.
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Model ID: alphanumeric, hyphens, underscores; 1–64 characters. This is the
# only caller string that renders into the published document, so the alphabet
# is deliberately inert — free text there would be caller-controlled output.
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Asset classes accepted by this template.
_VALID_ASSET_CLASSES = frozenset({"equity", "fixed_income", "fx", "commodity", "crypto", "multi_asset"})

# Maximum raw backtest data size (bytes) accepted inline.
_MAX_BACKTEST_DATA_BYTES = 10 * 1024 * 1024  # 10 MB

# Maximum number of per-period records accepted from the caller. A backtest of
# daily bars over the maximum 60-month window is ~1,260 records; the cap leaves
# headroom for intraday sampling without admitting an unbounded series. The
# adapter's byte cap on the structured parameters binds first for wide records,
# so this is the guard that stays meaningful for narrow ones.
_MAX_BACKTEST_RECORDS = 5_000

# Bounds for caller-supplied per-period figures. A simple return outside
# [-1000, 1000] (i.e. -100,000% .. +100,000%) is not a return series.
_RETURN_BOUND = 1000.0

# Bounds for the caller-supplied out-of-sample window, in whole months.
_OOS_MONTHS_MIN = 0.0
_OOS_MONTHS_MAX = 600.0

# Per-record numeric fields, with the bounds each is held to.
_RECORD_NUMERIC_FIELDS: Tuple[Tuple[str, float, float], ...] = (
    ("portfolio_return", -_RETURN_BOUND, _RETURN_BOUND),
    ("benchmark_return", -_RETURN_BOUND, _RETURN_BOUND),
    ("cumulative_return", -_RETURN_BOUND, _RETURN_BOUND),
)


def _finite_in_range(value: Any, low: float, high: float) -> Optional[float]:
    """Parse an untrusted caller numeric, fail CLOSED.

    Accepts only a real int/float (bool excluded) that is FINITE and within
    [low, high]; returns None on any violation. Strings are rejected outright,
    numeric strings included — the request contract is JSON numbers.

    NaN and +-Infinity are rejected explicitly. Both parse fine through
    ``float()`` and both survive a raw JSON document intact, and every
    comparison against NaN is False — so an unvalidated NaN return would make
    each threshold check silently pass as "not violated" instead of failing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if not low <= number <= high:
        return None
    return number


def _iso_date(value: Any) -> Optional[str]:
    """Return a normalised YYYY-MM-DD string, or None when it is not one."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _validate_records(raw: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Validate a caller-supplied backtest record series.

    Returns ``(records, None)`` on success or ``(None, reason)`` on rejection.
    The reason names the offending field and record index only — never the
    rejected value, which is caller-controlled text.
    """
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return None, "'backtest_records' must be a list of period records"
    if len(raw) > _MAX_BACKTEST_RECORDS:
        return None, (f"'backtest_records' exceeds the maximum of {_MAX_BACKTEST_RECORDS} entries")

    records: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return None, f"'backtest_records[{index}]' must be an object"

        record: Dict[str, Any] = {}

        record_date = _iso_date(entry.get("date"))
        if entry.get("date") is not None and record_date is None:
            return None, f"'backtest_records[{index}].date' must be a YYYY-MM-DD date"
        if record_date is not None:
            record["date"] = record_date

        for field, low, high in _RECORD_NUMERIC_FIELDS:
            if field not in entry or entry[field] is None:
                continue
            parsed = _finite_in_range(entry[field], low, high)
            if parsed is None:
                return None, (
                    f"'backtest_records[{index}].{field}' must be a finite number " f"between {low} and {high}"
                )
            record[field] = parsed

        if "portfolio_return" not in record:
            return None, (f"'backtest_records[{index}].portfolio_return' is required " "for every period record")
        records.append(record)

    return records, None


class PreProcessNode(FunctionNode):
    """Input validation: validate and parse the backtest validation payload.

    Outer backbone pre_process slot for FIN-C2-051 (FinC2051Agent).
    Rejects malformed or unauthorised invocation payloads before the inner
    DomainWorkflowGraph runs.

    Input state keys:
        user_input: JSON string (or object) with keys:
            model_id              (required) inert model identifier
            backtest_data         (required) file reference or inline data string
            validation_date_start (optional) ISO date YYYY-MM-DD
            validation_date_end   (optional) ISO date YYYY-MM-DD; not in the future
            asset_class           (required) one of equity/fixed_income/fx/
                                  commodity/crypto/multi_asset
        input_context: structured invocation parameters (the recommended channel
            for the record series, which the string channel would have to carry
            as escaped JSON):
            backtest_records      (optional) list of period records, each
                                  {date, portfolio_return, benchmark_return,
                                   cumulative_return}
            oos_months            (optional) out-of-sample window, whole months
            registration_date     (optional) ISO date the model was registered

    Output state keys (partial dict):
        validated_input: validated payload as a dict — consumed downstream by
                         FinC2051GraphNode.extract_input() and, across the
                         graph boundary, by BacktestIngestionNode. Both require
                         a dict (NOT a JSON string).
        status:          AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:       (on error only) list of error message strings
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _reject(self, reason: str) -> Dict[str, Any]:
        """Build the rejection delta. `reason` names a field, never a value."""
        logger.warning("PreProcessNode: input rejected — %s", reason)
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {reason}"],
            # The runner surfaces `formatted_output or result` as `output`. A reason left only in
            # error_log reaches no one: the terminal result carries just `status`, and get_output()
            # does not copy error_log out of the graph -- the caller sees a blank spinner.
            "formatted_output": "Request could not be completed. " + (f"PreProcessNode: {reason}"),
        }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and parse the backtest validation payload.

        Returns a PARTIAL dict: only the keys this node writes.
        """
        user_input = state.get("user_input", "")

        # --- Reject missing / non-string input ---
        if not user_input:
            return self._reject("'user_input' is empty or missing")

        if not isinstance(user_input, (str, dict)):
            return self._reject("'user_input' must be a JSON string or object")

        # --- Parse JSON if string ---
        if isinstance(user_input, str):
            stripped = user_input.strip()
            if not stripped:
                return self._reject("'user_input' is blank")
            try:
                payload: Dict[str, Any] = json.loads(stripped)
            except json.JSONDecodeError:
                return self._reject("'user_input' is not valid JSON")
        else:
            payload = dict(user_input)

        if not isinstance(payload, dict):
            return self._reject("'user_input' must be a JSON object")

        # --- Validate model_id (required) ---
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            return self._reject("'model_id' field is required")
        model_id = model_id.strip()
        if not _MODEL_ID_PATTERN.match(model_id):
            return self._reject(
                "'model_id' format is invalid (alphanumeric, hyphens and " "underscores, 1–64 characters)"
            )

        # --- Validate asset_class (required) ---
        asset_class = payload.get("asset_class")
        if not isinstance(asset_class, str) or not asset_class.strip():
            return self._reject("'asset_class' field is required")
        asset_class = asset_class.strip().lower()
        if asset_class not in _VALID_ASSET_CLASSES:
            return self._reject(f"'asset_class' must be one of {sorted(_VALID_ASSET_CLASSES)}")

        # --- Validate backtest_data (required) ---
        backtest_data = payload.get("backtest_data")
        if backtest_data is None:
            return self._reject("'backtest_data' field is required")
        backtest_data_str = str(backtest_data)
        if len(backtest_data_str.encode("utf-8")) > _MAX_BACKTEST_DATA_BYTES:
            return self._reject(
                f"'backtest_data' exceeds the maximum inline size of " f"{_MAX_BACKTEST_DATA_BYTES // (1024 * 1024)} MB"
            )

        # --- Validate validation_date_end (optional; must not be future) ---
        validation_date_end: Optional[str] = None
        raw_vde = payload.get("validation_date_end")
        if raw_vde is not None:
            validation_date_end = _iso_date(raw_vde)
            if validation_date_end is None:
                return self._reject("'validation_date_end' must be a YYYY-MM-DD date")
            if date.fromisoformat(validation_date_end) > date.today():
                return self._reject("'validation_date_end' must not be in the future")

        # --- validation_date_start (optional) ---
        validation_date_start: Optional[str] = None
        raw_vds = payload.get("validation_date_start")
        if raw_vds is not None:
            validation_date_start = _iso_date(raw_vds)
            if validation_date_start is None:
                return self._reject("'validation_date_start' must be a YYYY-MM-DD date")

        if (
            validation_date_start is not None
            and validation_date_end is not None
            and validation_date_start > validation_date_end
        ):
            return self._reject("'validation_date_start' must not be later than 'validation_date_end'")

        # --- Structured invocation parameters ---
        # Undeclared keys are ignored; every declared key is validated here.
        context = state.get("input_context")
        if context is not None and not isinstance(context, dict):
            return self._reject("'input_context' must be an object")
        context = context or {}

        records, records_error = _validate_records(context.get("backtest_records"))
        if records_error is not None:
            return self._reject(records_error)

        oos_months: Optional[float] = None
        if context.get("oos_months") is not None:
            oos_months = _finite_in_range(context.get("oos_months"), _OOS_MONTHS_MIN, _OOS_MONTHS_MAX)
            if oos_months is None:
                return self._reject(
                    f"'oos_months' must be a finite number between "
                    f"{int(_OOS_MONTHS_MIN)} and {int(_OOS_MONTHS_MAX)}"
                )

        registration_date: Optional[str] = None
        if context.get("registration_date") is not None:
            registration_date = _iso_date(context.get("registration_date"))
            if registration_date is None:
                return self._reject("'registration_date' must be a YYYY-MM-DD date")

        # Build the validated payload as a dict. The validation window is mapped
        # onto start_date / end_date — the keys BacktestIngestionNode reads —
        # and onto `period` / `validation_date`, which the generated document
        # renders, so the report header reflects the window that was requested.
        period = (
            f"{validation_date_start}/{validation_date_end}"
            if validation_date_start and validation_date_end
            else (validation_date_start or validation_date_end or "")
        )
        validated_payload: Dict[str, Any] = {
            "model_id": model_id,
            "asset_class": asset_class,
            "backtest_data": backtest_data_str,
            "validation_date_start": validation_date_start,
            "validation_date_end": validation_date_end,
            "start_date": validation_date_start,
            "end_date": validation_date_end,
            "period": period,
            "validation_date": validation_date_end or "",
            "backtest_records": records or [],
        }
        if oos_months is not None:
            validated_payload["oos_months"] = int(oos_months)
        if registration_date is not None:
            validated_payload["registration_date"] = registration_date

        logger.info(
            "PreProcessNode: validated model_id=%s asset_class=%s records=%d",
            model_id,
            asset_class,
            len(records or []),
        )

        # Audit trail: record validation completion (free-function form).
        emit_trace_event(
            "pre_process_validated",
            {
                "model_id": model_id,
                "asset_class": asset_class,
                "record_count": len(records or []),
            },
            state,
        )

        # validated_input is returned as a dict (NOT json.dumps()):
        # FinC2051GraphNode.extract_input() requires isinstance(validated, dict)
        # and BacktestIngestionNode.execute() requires isinstance(validated_input,
        # dict); a JSON string fails both checks, the inner graph never receives
        # the payload, and BacktestIngestionNode returns ERROR. State types
        # validated_input as Optional[Dict[str, Any]], so the dict already
        # matches the schema.
        return {
            "validated_input": validated_payload,
            "status": AgentStatus.SUCCESS.value,
        }
