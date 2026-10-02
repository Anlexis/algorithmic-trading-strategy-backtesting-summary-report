"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# FIN-C2-051 — Algorithmic Trading Strategy Backtesting Summary Report
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# Security note: no API keys, JWT tokens, or personal data in State.
# Credentials are accessed via config["configurable"] only (InvocationContext).
# llm_response is retained for the output gate's credential scan in
# PostProcessNode; it is not surfaced to the caller.

from typing import Any, Dict, List, Optional

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Flat TypedDict for FIN-C2-051 Algorithmic Trading Strategy Backtesting Summary Report.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode / BacktestReportGraphNode.merge_output
    # ------------------------------------------------------------------

    # Validated invocation payload produced by PreProcessNode.
    # Contains: {"strategy_id": str, "period": "YYYY-MM-DD/YYYY-MM-DD",
    #            "asset_class": str, "benchmark": str (optional)}
    # Raw user_input is NOT persisted beyond PreProcessNode.
    validated_input: Optional[Dict[str, Any]]

    # Final FSA-format backtesting validation document.
    # Written by merge_output() from inner graph's validation_document output.
    # Also mapped to state["result"] for PostProcessNode compatibility.
    result: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # BacktestIngestionNode output
    # Raw backtest result records for the target strategy and period.
    # Each record: {"date": str, "pnl": float, "cumulative_return": float,
    #               "position": float, "trade_count": int}
    backtest_results: Optional[List[Dict[str, Any]]]

    # BacktestIngestionNode output
    # Model / strategy metadata describing the backtested algorithm.
    # Keys: model_name (str), model_version (str), asset_class (str),
    #       strategy_type (str), start_date (str), end_date (str)
    model_metadata: Optional[Dict[str, Any]]

    # BacktestIngestionNode output
    # Benchmark return series for comparison (e.g. Nikkei 225, TOPIX).
    # Each record: {"date": str, "benchmark_return": float,
    #               "benchmark_cumulative_return": float}
    benchmark_data: Optional[List[Dict[str, Any]]]

    # ValidationMetricsNode output
    # Computed performance metrics derived from backtest_results.
    # Keys: sharpe_ratio (float), max_drawdown (float), calmar_ratio (float),
    #       sortino_ratio (float), annualized_return (float), win_rate (float)
    performance_metrics: Optional[Dict[str, Any]]

    # ValidationMetricsNode output
    # Computed risk metrics derived from backtest_results.
    # Keys: VaR_95 (float), CVaR_95 (float), volatility (float),
    #       beta (float), drawdown_periods (list of dicts)
    risk_metrics: Optional[Dict[str, Any]]

    # ValidationMetricsNode output
    # FSA compliance check results for each regulatory criterion.
    # Each entry: {"check_id": str, "check_name": str,
    #              "result": "PASS" | "FAIL" | "WARN", "detail": str}
    fsa_checks: Optional[List[Dict[str, Any]]]

    # DocumentGenerationNode output
    # Generated FSA-format validation document in markdown.
    # This field must pass the output gate before it reaches the caller.
    validation_document: Optional[str]

    # DocumentGenerationNode output
    # The rendered narrative, kept for the audit trail and scanned by the
    # output gate in PostProcessNode.
    llm_response: Optional[str]

    # DocumentGenerationNode output
    # Document metadata for audit trail and FSA submission tracking.
    # Keys: model_id (str), generated_at (str), validation_date (str),
    #       fsa_check_pass_count (int), fsa_check_fail_count (int)
    document_metadata: Optional[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
