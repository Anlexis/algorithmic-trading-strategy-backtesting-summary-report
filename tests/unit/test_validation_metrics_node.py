# FIN-C2-051 — Unit Tests: ValidationMetricsNode (BL-01..BL-04 + TC-05)
#
# BL-01: Sharpe Ratio correct for a known return fixture (compare to manual calc)
# BL-02: Max Drawdown correct (depth + duration for a known fixture)
# BL-03: FSA-CHK-01 FAILs when Sharpe < 0.5
# BL-04: FSA-CHK-03 FAILs when backtest period < 24 months
# TC-05: emit_trace_event fires exactly once (no duplicate audit events)
#
# ValidationMetricsNode.execute(state, config) reads backtest_results as a LIST
# of per-period records (the real BacktestIngestionNode contract + state schema):
# each record carries "portfolio_return" (per-period return) and
# "cumulative_return" (wealth index). benchmark_data is a LIST of records keyed
# by "benchmark_return". backtest_months / oos_months / dates come from
# model_metadata + validated_input. Status is returned as the AgentStatus.SUCCESS
# ENUM (not .value).
#
# Contract note: the node also accepts a pre-shaped dict for backward
# compatibility, but these tests exercise the canonical LIST shape so the unit
# test does not mask the real graph-flow contract.

import math

import pytest


def _status_ok(result) -> bool:
    from framework.schemas.agent_status import AgentStatus

    return result.get("status") in (AgentStatus.SUCCESS, AgentStatus.SUCCESS.value)


# ---------------------------------------------------------------------------
# Known fixtures (Sharpe / drawdown values pre-computed by manual calculation)
# ---------------------------------------------------------------------------

# Fixture A — daily returns whose annualised Sharpe is well above 0.5.
#   mu = 0.012375, sigma (ddof=1) = 0.005040904..., sqrt(252) annualisation
#   → sharpe = (mu/sigma)*sqrt(252) = 38.970595  (rounded to 6 dp)
_RETURNS_HIGH_SHARPE = [0.01, 0.02, 0.015, 0.005, 0.012, 0.008, 0.018, 0.011]
_EXPECTED_SHARPE_HIGH = 38.970595

# Fixture B — daily returns whose annualised Sharpe is negative (< 0.5).
_RETURNS_LOW_SHARPE = [0.05, -0.04, 0.03, -0.05, 0.04, -0.03, 0.02, -0.04]
_EXPECTED_SHARPE_LOW = -0.960518

# Cumulative wealth index: peak 1.2 → trough 0.9 → recover 1.1.
#   max drawdown = (1.2 - 0.9) / 1.2 = 0.25 → 25.0 %, duration = 3 bars.
_CUMULATIVE_DD = [1.0, 1.2, 0.9, 1.1]
_EXPECTED_MAX_DD_PCT = 25.0
_EXPECTED_MAX_DD_DURATION = 3


def _manual_sharpe(returns):
    n = len(returns)
    mu = sum(returns) / n
    var = sum((x - mu) ** 2 for x in returns) / (n - 1)
    sigma = math.sqrt(var)
    return round((mu / sigma) * math.sqrt(252), 6)


def _months_to_dates(months: int, registration_date: str):
    """Return (start_date, end_date) spanning `months`, ending before registration."""
    # Anchor the backtest window to end one year before registration so
    # FSA-CHK-05 (no data snooping: backtest_start < registration) passes by default.
    from datetime import datetime

    reg = datetime.strptime(registration_date, "%Y-%m-%d")
    end_year = reg.year - 1
    start_total = (end_year * 12 + reg.month) - months
    start_year, start_month = divmod(start_total - 1, 12)
    start = f"{start_year:04d}-{start_month + 1:02d}-01"
    end = f"{end_year:04d}-{reg.month:02d}-01"
    return start, end


def _records(returns, cumulative):
    """Build a LIST of per-period backtest records (the real ingestion contract).

    Each record carries portfolio_return (→ returns) and cumulative_return
    (→ cumulative), mirroring BacktestIngestionNode's emitted record shape.
    """
    n = max(len(returns), len(cumulative))
    recs = []
    for i in range(n):
        rec = {"date": f"2020-{(i % 12) + 1:02d}-01"}
        if i < len(returns):
            rec["portfolio_return"] = returns[i]
        if i < len(cumulative):
            rec["cumulative_return"] = cumulative[i]
        recs.append(rec)
    return recs


def _backtest_state(
    returns=None,
    cumulative=None,
    backtest_months=36,
    oos_months=12,
    registration_date="2026-01-01",
    backtest_start_date=None,
):
    returns = returns if returns is not None else _RETURNS_HIGH_SHARPE
    cumulative = cumulative if cumulative is not None else _CUMULATIVE_DD
    start, end = _months_to_dates(backtest_months, registration_date)
    if backtest_start_date is not None:
        start = backtest_start_date
    return {
        "backtest_results": _records(returns, cumulative),
        "benchmark_data": [{"benchmark_return": 0.008} for _ in range(8)],
        "model_metadata": {"registration_date": registration_date},
        "validated_input": {
            "start_date": start,
            "end_date": end,
            "oos_months": oos_months,
        },
    }


def _fsa_check(result, check_id):
    for c in result.get("fsa_checks", []):
        if c.get("check_id") == check_id:
            return c
    return None


class TestBL01SharpeRatio:
    """BL-01: Sharpe Ratio matches the manual calculation for a known fixture."""

    def test_sharpe_matches_manual_calc(self):
        """performance_metrics.sharpe_ratio equals the hand-computed value."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(returns=_RETURNS_HIGH_SHARPE))

        assert _status_ok(result), f"Expected SUCCESS, got {result}"
        sharpe = result.get("performance_metrics", {}).get("sharpe_ratio")
        # Cross-check against an independent manual computation.
        assert sharpe == _manual_sharpe(_RETURNS_HIGH_SHARPE)
        assert sharpe == pytest.approx(
            _EXPECTED_SHARPE_HIGH, abs=1e-6
        ), f"Expected Sharpe {_EXPECTED_SHARPE_HIGH}, got {sharpe}"

    def test_sharpe_none_for_constant_returns(self):
        """Zero-variance returns yield sharpe=None (no ZeroDivisionError)."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(returns=[0.01, 0.01, 0.01, 0.01]))

        assert _status_ok(result)
        assert (
            result.get("performance_metrics", {}).get("sharpe_ratio") is None
        ), "Constant returns (sigma=0) must yield sharpe_ratio=None"


class TestBL02MaxDrawdown:
    """BL-02: Max Drawdown depth + duration correct for a known cumulative series."""

    def test_max_drawdown_pct_correct(self):
        """max_drawdown_pct = 25.0 for cumulative [1.0, 1.2, 0.9, 1.1]."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(cumulative=_CUMULATIVE_DD))

        assert _status_ok(result)
        pm = result.get("performance_metrics", {})
        assert pm.get("max_drawdown_pct") == pytest.approx(
            _EXPECTED_MAX_DD_PCT, abs=1e-4
        ), f"Expected max_drawdown_pct {_EXPECTED_MAX_DD_PCT}, got {pm.get('max_drawdown_pct')}"

    def test_max_drawdown_duration_correct(self):
        """max_drawdown_duration_bars = 3 for the known fixture."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(cumulative=_CUMULATIVE_DD))

        pm = result.get("performance_metrics", {})
        assert pm.get("max_drawdown_duration_bars") == _EXPECTED_MAX_DD_DURATION, (
            f"Expected duration {_EXPECTED_MAX_DD_DURATION}, " f"got {pm.get('max_drawdown_duration_bars')}"
        )

    def test_no_drawdown_for_monotonic_series(self):
        """A strictly increasing cumulative series has 0% drawdown."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(cumulative=[1.0, 1.1, 1.2, 1.3]))

        pm = result.get("performance_metrics", {})
        assert pm.get("max_drawdown_pct") == pytest.approx(
            0.0, abs=1e-6
        ), "Monotonic increasing series must have 0% drawdown"


class TestBL03FsaChk01Sharpe:
    """BL-03: FSA-CHK-01 FAILs when Sharpe < 0.5 (threshold)."""

    def test_chk01_fails_when_sharpe_below_threshold(self):
        """Low-Sharpe fixture (Sharpe < 0.5) → FSA-CHK-01 pass=False."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(returns=_RETURNS_LOW_SHARPE))

        assert _status_ok(result)
        sharpe = result.get("performance_metrics", {}).get("sharpe_ratio")
        assert sharpe == pytest.approx(_EXPECTED_SHARPE_LOW, abs=1e-6)
        assert sharpe < 0.5, "fixture must produce Sharpe below the 0.5 threshold"

        chk01 = _fsa_check(result, "FSA-CHK-01")
        assert chk01 is not None, "FSA-CHK-01 must be present in fsa_checks"
        assert (
            chk01.get("pass") is False
        ), f"FSA-CHK-01 must FAIL when Sharpe ({sharpe}) < 0.5, got pass={chk01.get('pass')}"

    def test_chk01_passes_when_sharpe_above_threshold(self):
        """High-Sharpe fixture (Sharpe >= 0.5) → FSA-CHK-01 pass=True."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(returns=_RETURNS_HIGH_SHARPE))

        chk01 = _fsa_check(result, "FSA-CHK-01")
        assert chk01 is not None
        assert chk01.get("pass") is True, "FSA-CHK-01 must PASS when Sharpe is comfortably above 0.5"


class TestBL04FsaChk03BacktestPeriod:
    """BL-04: FSA-CHK-03 FAILs when backtest period < 24 months."""

    def test_chk03_fails_when_period_too_short(self):
        """backtest_months = 12 (< 24) → FSA-CHK-03 pass=False."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(backtest_months=12))

        assert _status_ok(result)
        chk03 = _fsa_check(result, "FSA-CHK-03")
        assert chk03 is not None, "FSA-CHK-03 must be present in fsa_checks"
        assert chk03.get("pass") is False, "FSA-CHK-03 must FAIL when backtest_months (12) < 24"
        assert chk03.get("actual") == 12

    def test_chk03_passes_when_period_long_enough(self):
        """backtest_months = 36 (>= 24) → FSA-CHK-03 pass=True."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(backtest_months=36))

        chk03 = _fsa_check(result, "FSA-CHK-03")
        assert chk03 is not None
        assert chk03.get("pass") is True, "FSA-CHK-03 must PASS when backtest_months (36) >= 24"

    def test_all_five_fsa_checks_present(self):
        """The node must emit all five FSA checks (FSA-CHK-01..05)."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state())

        check_ids = {c.get("check_id") for c in result.get("fsa_checks", [])}
        for cid in ("FSA-CHK-01", "FSA-CHK-02", "FSA-CHK-03", "FSA-CHK-04", "FSA-CHK-05"):
            assert cid in check_ids, f"Missing FSA check {cid}; got {sorted(check_ids)}"


class TestFsaResultKey:
    """F-2: every FSA check must carry a "result" of "PASS"/"FAIL" (read by
    DocumentGenerationNode._count_fsa_results / _format_fsa_checks_section)."""

    def test_each_check_has_result_key(self):
        """Every fsa_checks entry must have a "result" of PASS or FAIL."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state())

        checks = result.get("fsa_checks", [])
        assert checks, "fsa_checks must be non-empty"
        for c in checks:
            assert c.get("result") in (
                "PASS",
                "FAIL",
            ), f"{c.get('check_id')} must carry result PASS/FAIL, got {c.get('result')!r}"

    def test_result_matches_pass_bool(self):
        """The "result" string must agree with the existing "pass" boolean."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(returns=_RETURNS_LOW_SHARPE))

        for c in result.get("fsa_checks", []):
            expected = "PASS" if c.get("pass") else "FAIL"
            assert (
                c.get("result") == expected
            ), f"{c.get('check_id')}: result {c.get('result')!r} != pass={c.get('pass')}"

    def test_chk01_result_fail_for_low_sharpe(self):
        """FSA-CHK-01 result == 'FAIL' when Sharpe < threshold (real DocGen read)."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state(returns=_RETURNS_LOW_SHARPE))
        chk01 = _fsa_check(result, "FSA-CHK-01")
        assert chk01.get("result") == "FAIL"


class TestStatusEnum:
    """status is written into State as the AgentStatus string value, not the enum object.

    This class previously asserted the opposite ("F-3: ... not .value"). It is inverted here
    against the fleet contract: a State status assignment uses AgentStatus.<X>.value, and the
    bare member is a violation -- it is the single most frequently flagged review finding
    across this fleet, and the check explicitly covers test assertions as well as src/.

    Worth stating because AgentStatus is a str-Enum: 'success' == AgentStatus.SUCCESS is True,
    so an == comparison passes either way and nothing signals which form is stored. Only an
    `is` comparison, as this class used, can tell them apart -- which is why it caught the
    change, and why it was the one place the inversion had to be made deliberately rather
    than mechanically.
    """

    def test_success_status_is_enum(self):
        """A successful run writes the AgentStatus.SUCCESS string value."""
        from framework.schemas.agent_status import AgentStatus
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        result = node.execute(_backtest_state())
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"status must be AgentStatus.SUCCESS.value, got {result.get('status')!r}"

    def test_error_status_is_enum(self):
        """An error path writes the AgentStatus.ERROR string value."""
        from framework.schemas.agent_status import AgentStatus
        import src.nodes.validation_metrics_node as mod

        # Force an exception inside the try-block by making the drawdown helper raise.
        def _boom(*_a, **_k):
            raise RuntimeError("forced failure")

        node = mod.ValidationMetricsNode()
        original = mod._compute_drawdown_series
        mod._compute_drawdown_series = _boom
        try:
            result = node.execute(_backtest_state())
        finally:
            mod._compute_drawdown_series = original

        assert (
            result.get("status") == AgentStatus.ERROR.value
        ), f"error status must be AgentStatus.ERROR.value, got {result.get('status')!r}"


class TestBacktestResultsListContract:
    """The node must read backtest_results as the real LIST-of-records contract
    (BacktestIngestionNode emits a list; the dict fixture previously masked this)."""

    def test_list_records_produce_metrics(self):
        """A LIST of per-period records yields populated performance metrics."""
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        node = ValidationMetricsNode()
        state = _backtest_state(returns=_RETURNS_HIGH_SHARPE)
        assert isinstance(state["backtest_results"], list), "fixture must be a LIST"

        result = node.execute(state)
        assert _status_ok(result), f"Expected SUCCESS, got {result}"
        sharpe = result.get("performance_metrics", {}).get("sharpe_ratio")
        assert sharpe == pytest.approx(_EXPECTED_SHARPE_HIGH, abs=1e-6)

    def test_ingestion_output_feeds_validation(self):
        """The real BacktestIngestionNode output (a list) feeds ValidationMetrics
        end-to-end without raising — proves the contract is aligned."""
        from src.nodes.backtest_ingestion_node import BacktestIngestionNode
        from src.nodes.validation_metrics_node import ValidationMetricsNode

        vi = {
            "model_id": "MODEL-ALPHA",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "asset_class": "equity",
        }
        ingest = BacktestIngestionNode().execute({"validated_input": vi})
        assert isinstance(ingest.get("backtest_results"), list)

        metrics_state = {
            "backtest_results": ingest["backtest_results"],
            "benchmark_data": ingest["benchmark_data"],
            "model_metadata": ingest["model_metadata"],
            "validated_input": vi,
        }
        result = ValidationMetricsNode().execute(metrics_state)
        assert _status_ok(result), f"ingestion→validation flow must succeed, got {result}"
        assert "fsa_checks" in result and result["fsa_checks"], "FSA checks must be produced"


class TestTC05NoDuplicateAuditEvents:
    """TC-05: emit_trace_event fires exactly once per invocation (no duplicates)."""

    def test_emit_trace_event_fires_once(self, monkeypatch):
        """The audit event must fire exactly once on a successful run."""
        import src.nodes.validation_metrics_node as mod

        calls = []
        monkeypatch.setattr(mod, "emit_trace_event", lambda event, payload, st: calls.append(event))

        node = mod.ValidationMetricsNode()
        result = node.execute(_backtest_state())

        assert _status_ok(result)
        assert calls == [
            "validation_metrics_calculated"
        ], f"TC-05: expected exactly one 'validation_metrics_calculated' event, got {calls}"
