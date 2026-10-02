"""AgentCore Platform v1.0"""

# Node contract (agents_layer_design.md §1):
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings [A1]
#  - Never import from the Level-0 platform SDK (import isolation gate)
#
# FIN-C2-051 — ValidationMetricsNode
# Domain node: compute quantitative performance/risk metrics from backtesting
# results and run FSA model validation checks.
#
# Performance metrics: Sharpe, Sortino, Calmar, annualized return, win rate,
#                       max drawdown (absolute + duration)
# Risk metrics:        VaR/CVaR 95%, annualized volatility, beta, drawdown periods
# FSA checks:          FSA-CHK-01..05 (configurable thresholds)
#
# Input:  state["backtest_results"], state["benchmark_data"], state["model_metadata"]
# Output: state["performance_metrics"], state["risk_metrics"], state["fsa_checks"]

import logging
import math
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail — free-function form only.
# Use the bare emit_trace_event() call, not a method on self (AttributeError).
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Annualisation factor: assume daily returns → 252 trading days per year.
_TRADING_DAYS_PER_YEAR = 252

# Confidence level for VaR/CVaR.
_VAR_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Pure-Python helpers (no external numeric library required)
# ---------------------------------------------------------------------------


def _mean(values: List[float]) -> Optional[float]:
    """Return arithmetic mean, or None for an empty list."""
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: List[float], ddof: int = 1) -> Optional[float]:
    """Return sample standard deviation (ddof=1), or None for < 2 values."""
    n = len(values)
    if n < max(2, ddof + 1):
        return None
    mu = sum(values) / n
    variance = sum((x - mu) ** 2 for x in values) / (n - ddof)
    return math.sqrt(variance)


def _downside_std(values: List[float], threshold: float = 0.0) -> Optional[float]:
    """Return downside deviation (semi-std below threshold)."""
    below = [x - threshold for x in values if x < threshold]
    if len(below) < 2:
        return None
    variance = sum(x**2 for x in below) / len(below)
    return math.sqrt(variance)


def _compute_drawdown_series(
    cumulative_returns: List[float],
) -> Tuple[float, int, List[Dict[str, Any]]]:
    """Compute max drawdown (as positive fraction), its duration in bars, and
    all drawdown periods.

    Returns (max_drawdown, max_duration_bars, drawdown_periods).
    max_drawdown is expressed as a positive value (e.g. 0.25 = 25% drawdown).
    """
    if not cumulative_returns:
        return (0.0, 0, [])

    peak = cumulative_returns[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_duration = 0
    periods: List[Dict[str, Any]] = []
    in_drawdown = False
    dd_start = 0

    for i, val in enumerate(cumulative_returns):
        if val >= peak:
            if in_drawdown:
                dd = (peak - min(cumulative_returns[dd_start:i])) / (abs(peak) or 1.0)
                duration = i - dd_start
                periods.append({"start": dd_start, "end": i - 1, "depth": round(dd, 6)})
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = duration
                in_drawdown = False
            peak = val
            peak_idx = i
        else:
            if not in_drawdown:
                in_drawdown = True
                dd_start = peak_idx

    # Handle open drawdown at end of series.
    if in_drawdown:
        dd = (peak - min(cumulative_returns[dd_start:])) / (abs(peak) or 1.0)
        duration = len(cumulative_returns) - dd_start
        periods.append({"start": dd_start, "end": len(cumulative_returns) - 1, "depth": round(dd, 6)})
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = duration

    return (round(max_dd, 6), max_dd_duration, periods)


def _compute_var_cvar(
    returns: List[float], confidence: float = _VAR_CONFIDENCE
) -> Tuple[Optional[float], Optional[float]]:
    """Compute historical VaR and CVaR at the given confidence level.

    Returns (VaR, CVaR) as positive values representing loss magnitude,
    or (None, None) if returns is empty.
    """
    if not returns:
        return (None, None)
    sorted_r = sorted(returns)
    idx = int(math.floor((1.0 - confidence) * len(sorted_r)))
    idx = max(0, idx)
    var = -sorted_r[idx]  # Positive loss value.
    tail = sorted_r[: idx + 1] if idx + 1 <= len(sorted_r) else sorted_r
    cvar = -(_mean(tail) or 0.0)
    return (round(var, 6), round(cvar, 6))


def _annualized_return(total_return: float, n_bars: int, bars_per_year: int) -> Optional[float]:
    """Convert a total return fraction to an annualised return.

    Returns None when n_bars is 0.
    """
    if n_bars <= 0 or bars_per_year <= 0:
        return None
    years = n_bars / bars_per_year
    if years <= 0:
        return None
    try:
        ann = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    except (ValueError, ZeroDivisionError):
        return None
    return round(ann, 6)


def _records_to_returns(records: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[float]:
    """Extract a per-period return series from a list of backtest/benchmark records.

    Reads the first key present (in priority order) from each record, coercing
    to float. Records without any of the keys are skipped.
    """
    out: List[float] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for k in keys:
            if k in rec and rec[k] is not None:
                try:
                    out.append(float(rec[k]))
                except (TypeError, ValueError):
                    pass
                break
    return out


def _cumulative_from_returns(returns: List[float]) -> List[float]:
    """Build a cumulative wealth index (starting at 1.0) by compounding returns.

    Used when the backtest records carry per-period returns but no explicit
    cumulative series.
    """
    cumulative: List[float] = []
    wealth = 1.0
    for r in returns:
        wealth *= 1.0 + r
        cumulative.append(round(wealth, 8))
    return cumulative


def _adapt_backtest_results(
    backtest_results: Any,
    benchmark_data: Any,
    model_meta: Dict[str, Any],
    validated_input: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalise the backtest/benchmark inputs into the metric-input dict shape.

    Contract (src/schemas/state.py + BacktestIngestionNode): backtest_results is
    a LIST of per-period records and benchmark_data is a LIST of benchmark
    records. BacktestIngestionNode emits records keyed by "portfolio_return"
    (per-period return) / "benchmark_return"; the state-schema record shape also
    allows "pnl" / "cumulative_return". This adapter reads what the upstream node
    actually emits and derives the series ValidationMetricsNode needs.

    For backward compatibility the function also accepts a pre-shaped DICT (the
    legacy unit-test fixture form) and passes its fields through unchanged.
    """
    # Legacy / test path: backtest_results already a pre-shaped metrics dict.
    if isinstance(backtest_results, dict):
        bench_returns: List[float]
        if isinstance(benchmark_data, dict):
            bench_returns = [float(r) for r in (benchmark_data.get("returns") or [])]
        else:
            bench_returns = _records_to_returns(benchmark_data or [], ("benchmark_return", "return"))
        return {
            "returns": [float(r) for r in (backtest_results.get("returns") or [])],
            "cumulative": [float(c) for c in (backtest_results.get("cumulative") or [])],
            "bench_returns": bench_returns,
            "win_count": int(backtest_results.get("win_count", 0)),
            "total_count": int(backtest_results.get("total_count", 0)),
            "backtest_months": int(backtest_results.get("backtest_months", 0)),
            "oos_months": int(backtest_results.get("oos_months", 0)),
            "registration_date": str(
                backtest_results.get("registration_date") or model_meta.get("registration_date") or ""
            ),
            "backtest_start_date": str(
                backtest_results.get("backtest_start_date") or model_meta.get("backtest_period_start") or ""
            ),
        }

    # Canonical path: backtest_results is a LIST of per-period records.
    records: List[Dict[str, Any]] = backtest_results if isinstance(backtest_results, list) else []
    returns = _records_to_returns(records, ("portfolio_return", "pnl", "return"))

    # Prefer an explicit cumulative series if the records carry one; otherwise
    # build a wealth index by compounding the per-period returns.
    cumulative = _records_to_returns(records, ("cumulative_return", "cumulative"))
    if not cumulative:
        cumulative = _cumulative_from_returns(returns)

    bench_records: List[Dict[str, Any]] = benchmark_data if isinstance(benchmark_data, list) else []
    bench_returns = _records_to_returns(bench_records, ("benchmark_return", "return"))

    win_count = sum(1 for r in returns if r > 0.0)
    total_count = len(returns)

    # Period months: derive from validated_input dates, else model_metadata.
    start = str(validated_input.get("start_date") or model_meta.get("backtest_period_start") or "")
    end = str(validated_input.get("end_date") or model_meta.get("backtest_period_end") or "")
    backtest_months = _months_between_dates(start, end)

    return {
        "returns": returns,
        "cumulative": cumulative,
        "bench_returns": bench_returns,
        "win_count": win_count,
        "total_count": total_count,
        "backtest_months": backtest_months,
        "oos_months": int(validated_input.get("oos_months") or model_meta.get("oos_months") or 0),
        "registration_date": str(model_meta.get("registration_date") or ""),
        "backtest_start_date": start or str(model_meta.get("backtest_period_start") or ""),
    }


def _months_between_dates(start_date: str, end_date: str) -> int:
    """Approximate whole months between two YYYY-MM-DD strings (0 on parse error)."""
    from datetime import datetime

    try:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d")
        return max(0, (e.year - s.year) * 12 + (e.month - s.month))
    except (ValueError, TypeError):
        return 0


def _compute_beta(asset_returns: List[float], benchmark_returns: List[float]) -> Optional[float]:
    """Compute beta of asset vs benchmark via covariance / variance.

    Returns None when there is insufficient data or benchmark variance is zero.
    """
    n = min(len(asset_returns), len(benchmark_returns))
    if n < 2:
        return None
    ar = asset_returns[:n]
    br = benchmark_returns[:n]
    mu_a = sum(ar) / n
    mu_b = sum(br) / n
    cov = sum((ar[i] - mu_a) * (br[i] - mu_b) for i in range(n)) / (n - 1)
    var_b = sum((br[i] - mu_b) ** 2 for i in range(n)) / (n - 1)
    if var_b == 0.0:
        return None
    return round(cov / var_b, 6)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class ValidationMetricsNode(FunctionNode):
    """Compute performance/risk metrics from backtesting results and run FSA checks.

    Domain node (registered as "validation_metrics" in DomainWorkflowGraph).

    Input state keys:
        backtest_results:  dict with keys:
            returns          list[float]  — per-bar simple returns
            cumulative       list[float]  — cumulative wealth index
            win_count        int          — number of profitable bars/trades
            total_count      int          — total bars/trades
            backtest_months  int          — length of the backtest window in months
            oos_months       int          — out-of-sample window in months (0 = absent)
            registration_date  str        — model registration ISO date (YYYY-MM-DD)
            backtest_start_date str       — backtest start ISO date (YYYY-MM-DD)
        benchmark_data:    dict with key:
            returns          list[float]  — benchmark per-bar simple returns
        model_metadata:    dict (arbitrary; checked for "registration_date" fallback)

    Output state keys (partial dict — only changed fields):
        performance_metrics:  dict  (Sharpe, Sortino, Calmar, ann_return, win_rate,
                                     max_drawdown, max_drawdown_duration)
        risk_metrics:         dict  (var_95, cvar_95, ann_volatility, beta, drawdown_periods)
        fsa_checks:           list[dict]  (one entry per FSA-CHK-0N)
        status:               AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:            list[str]  (on error only)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    # Built-in FSA thresholds, used when config/config.yaml declares no override.
    _DEFAULT_FSA_THRESHOLDS: ClassVar[Dict[str, Any]] = {
        "fsa_chk01_min_sharpe": 0.5,
        "fsa_chk02_max_drawdown_pct": 30.0,
        "fsa_chk03_min_backtest_months": 24,
        "fsa_chk04_min_oos_months": 6,
    }

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None) -> None:
        """Hold the effective FSA thresholds for this node instance.

        `thresholds` carries the values declared in config/config.yaml, already
        shape- and range-checked by the outer graph before they were forwarded
        (src/graph/graph.py). Anything absent keeps its built-in default.

        The thresholds are resolved here rather than from an ``execute(state,
        config)`` parameter: the framework calls ``execute(state)`` with one
        argument, so a second parameter is never populated and every value read
        from it would silently be the default.
        """
        super().__init__()
        self._thresholds: Dict[str, Any] = {
            **self._DEFAULT_FSA_THRESHOLDS,
            **(thresholds or {}),
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        thresholds: Dict[str, Any] = self._thresholds

        # ------------------------------------------------------------------
        # 1. Extract inputs (graceful defaults for missing/partial state).
        #
        # backtest_results is a LIST of per-period records (state-schema +
        # BacktestIngestionNode contract); _adapt_backtest_results() derives the
        # return/cumulative series and counts from it. A pre-shaped dict is also
        # accepted for backward compatibility.
        # ------------------------------------------------------------------
        model_meta: Dict[str, Any] = state.get("model_metadata") or {}
        validated_input: Dict[str, Any] = state.get("validated_input") or {}

        adapted = _adapt_backtest_results(
            state.get("backtest_results"),
            state.get("benchmark_data"),
            model_meta,
            validated_input,
        )

        returns: List[float] = adapted["returns"]
        cumulative: List[float] = adapted["cumulative"]
        bench_returns: List[float] = adapted["bench_returns"]

        win_count: int = adapted["win_count"]
        total_count: int = adapted["total_count"]
        backtest_months: int = adapted["backtest_months"]
        oos_months: int = adapted["oos_months"]
        registration_date: str = adapted["registration_date"]
        backtest_start_date: str = adapted["backtest_start_date"]

        try:
            # ------------------------------------------------------------------
            # 2. Performance metrics.
            # ------------------------------------------------------------------
            n = len(returns)
            mu = _mean(returns)
            sigma = _std(returns, ddof=1)

            # Sharpe Ratio (annualised, risk-free = 0).
            if mu is not None and sigma and sigma > 0.0:
                sharpe = round((mu / sigma) * math.sqrt(_TRADING_DAYS_PER_YEAR), 6)
            else:
                sharpe = None

            # Sortino Ratio (annualised, threshold = 0).
            down_sigma = _downside_std(returns, threshold=0.0)
            if mu is not None and down_sigma and down_sigma > 0.0:
                sortino = round((mu / down_sigma) * math.sqrt(_TRADING_DAYS_PER_YEAR), 6)
            else:
                sortino = None

            # Max drawdown.
            max_dd, max_dd_dur, drawdown_periods = _compute_drawdown_series(cumulative)
            max_dd_pct = round(max_dd * 100.0, 4)  # expressed as percentage

            # Calmar Ratio = annualised return / max drawdown.
            total_return = (
                (cumulative[-1] - cumulative[0]) / abs(cumulative[0])
                if (cumulative and abs(cumulative[0]) > 0)
                else None
            )
            ann_ret = _annualized_return(total_return, n, _TRADING_DAYS_PER_YEAR) if total_return is not None else None
            if ann_ret is not None and max_dd and max_dd > 0.0:
                calmar = round(ann_ret / max_dd, 6)
            else:
                calmar = None

            # Win rate.
            win_rate = round(win_count / total_count * 100.0, 4) if total_count > 0 else None

            performance_metrics: Dict[str, Any] = {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "annualized_return": ann_ret,
                "win_rate_pct": win_rate,
                "max_drawdown_pct": max_dd_pct,
                "max_drawdown_duration_bars": max_dd_dur,
            }

            # ------------------------------------------------------------------
            # 3. Risk metrics.
            # ------------------------------------------------------------------
            var_95, cvar_95 = _compute_var_cvar(returns, _VAR_CONFIDENCE)
            ann_vol = round(sigma * math.sqrt(_TRADING_DAYS_PER_YEAR), 6) if sigma is not None else None
            beta = _compute_beta(returns, bench_returns)

            risk_metrics: Dict[str, Any] = {
                "var_95": var_95,
                "cvar_95": cvar_95,
                "annualized_volatility": ann_vol,
                "beta": beta,
                "drawdown_periods": drawdown_periods,
            }

            # ------------------------------------------------------------------
            # 4. FSA compliance checks.
            # ------------------------------------------------------------------
            fsa_checks: List[Dict[str, Any]] = []

            # FSA-CHK-01: Sharpe Ratio >= min_sharpe.
            min_sharpe: float = float(thresholds["fsa_chk01_min_sharpe"])
            chk01_pass = sharpe is not None and sharpe >= min_sharpe
            fsa_checks.append(
                {
                    "check_id": "FSA-CHK-01",
                    "check_name": "Sharpe Ratio",
                    "description": f"Sharpe Ratio >= {min_sharpe}",
                    "threshold": min_sharpe,
                    "actual": sharpe,
                    "pass": chk01_pass,
                    "result": "PASS" if chk01_pass else "FAIL",
                    "detail": (f"Sharpe={sharpe:.4f}" if sharpe is not None else "Insufficient data for Sharpe"),
                }
            )

            # FSA-CHK-02: Max Drawdown <= max_drawdown_pct.
            max_dd_threshold: float = float(thresholds["fsa_chk02_max_drawdown_pct"])
            chk02_pass = max_dd_pct <= max_dd_threshold
            fsa_checks.append(
                {
                    "check_id": "FSA-CHK-02",
                    "check_name": "Max Drawdown",
                    "description": f"Max Drawdown <= {max_dd_threshold}%",
                    "threshold": max_dd_threshold,
                    "actual": max_dd_pct,
                    "pass": chk02_pass,
                    "result": "PASS" if chk02_pass else "FAIL",
                    "detail": f"MaxDD={max_dd_pct:.4f}%",
                }
            )

            # FSA-CHK-03: Backtest period >= min_backtest_months.
            min_bt_months: int = int(thresholds["fsa_chk03_min_backtest_months"])
            chk03_pass = backtest_months >= min_bt_months
            fsa_checks.append(
                {
                    "check_id": "FSA-CHK-03",
                    "check_name": "Backtest Period",
                    "description": f"Backtest period >= {min_bt_months} months",
                    "threshold": min_bt_months,
                    "actual": backtest_months,
                    "pass": chk03_pass,
                    "result": "PASS" if chk03_pass else "FAIL",
                    "detail": f"backtest_months={backtest_months}",
                }
            )

            # FSA-CHK-04: Out-of-sample period >= min_oos_months.
            min_oos_months: int = int(thresholds["fsa_chk04_min_oos_months"])
            chk04_pass = oos_months >= min_oos_months
            fsa_checks.append(
                {
                    "check_id": "FSA-CHK-04",
                    "check_name": "Out-of-sample Period",
                    "description": f"Out-of-sample period >= {min_oos_months} months",
                    "threshold": min_oos_months,
                    "actual": oos_months,
                    "pass": chk04_pass,
                    "result": "PASS" if chk04_pass else "FAIL",
                    "detail": (f"oos_months={oos_months}" if oos_months > 0 else "No out-of-sample period present"),
                }
            )

            # FSA-CHK-05: No data snooping (backtest_start_date < registration_date).
            if backtest_start_date and registration_date:
                chk05_pass = backtest_start_date < registration_date
                chk05_detail = (
                    f"backtest_start={backtest_start_date} < registration={registration_date}"
                    if chk05_pass
                    else (
                        f"Potential data snooping: backtest_start={backtest_start_date}"
                        f" >= registration={registration_date}"
                    )
                )
            else:
                chk05_pass = False
                chk05_detail = "Missing backtest_start_date or registration_date — cannot verify"
            fsa_checks.append(
                {
                    "check_id": "FSA-CHK-05",
                    "check_name": "No Data Snooping",
                    "description": "No data snooping (backtest period predates model registration)",
                    "threshold": None,
                    "actual": {
                        "backtest_start_date": backtest_start_date,
                        "registration_date": registration_date,
                    },
                    "pass": chk05_pass,
                    "result": "PASS" if chk05_pass else "FAIL",
                    "detail": chk05_detail,
                }
            )

            fsa_pass = [c["check_id"] for c in fsa_checks if c["pass"]]
            fsa_fail = [c["check_id"] for c in fsa_checks if not c["pass"]]

            logger.info(
                "ValidationMetricsNode: sharpe=%s max_dd_pct=%s fsa_pass=%s fsa_fail=%s",
                sharpe,
                max_dd_pct,
                fsa_pass,
                fsa_fail,
            )

            # ------------------------------------------------------------------
            # 5. Audit trail — free-function form (not a node method call).
            # ------------------------------------------------------------------
            emit_trace_event(
                "validation_metrics_calculated",
                {
                    "fsa_pass": fsa_pass,
                    "fsa_fail": fsa_fail,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd_pct,
                },
                state,
            )

            return {
                "performance_metrics": performance_metrics,
                "risk_metrics": risk_metrics,
                "fsa_checks": fsa_checks,
                "status": AgentStatus.SUCCESS.value,
            }

        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("ValidationMetricsNode: unexpected error: %s", exc)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"ValidationMetricsNode: {exc}"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. " + (f"ValidationMetricsNode: {exc}"),
            }
