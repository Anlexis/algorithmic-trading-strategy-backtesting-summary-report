"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from api/ or from another agent
#
# FIN-C2-051 — DocumentGenerationNode
# Inner domain graph node: assemble the model validation document from the
# computed metrics and the regulatory check results already in state.
#
# Input:  state["performance_metrics"], state["risk_metrics"],
#         state["fsa_checks"], state["model_metadata"],
#         state["validated_input"]
# Output: state["validation_document"], state["llm_response"],
#         state["result"], state["document_metadata"]
#
# Generation mode: deterministic. Every sentence in the document is rendered
# from structured numeric state by the functions in this module — there is no
# model call anywhere in the path, and config/agent.yaml declares
# generation_mode: deterministic to match. The narrative field kept for the
# audit trail (llm_response) carries that same rendered text.

import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security.credential_detector import detect_credentials_in_value

# Audit trail — the free-function form; the instance-method form does not exist.
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Output-bearing fields this node writes. The output gate clears every one of
# them on a violation, and an inventory assertion in the gate fails if a new
# output field is added to execute() without being listed here — so a future
# field cannot quietly escape the clearing.
_OUTPUT_BEARING_FIELDS: Tuple[str, ...] = (
    "validation_document",
    "llm_response",
    "result",
    "document_metadata",
)

# The notice that replaces the document when the gate withholds it. It must be
# TRUTHY: the framework projects `formatted_output or result`, so an empty
# notice would re-activate the fallback onto whatever `result` still holds.
_WITHHELD_NOTICE = (
    "The validation document was withheld: the output gate found content that "
    "must not leave the agent. No document is available for this request."
)


def _cell(metrics: Dict[str, Any], key: str) -> str:
    """Render one metric cell.

    ``dict.get(key, "N/A")`` is not enough: a metric that could not be computed
    is PRESENT with the value None, so the default never applies and the table
    renders the literal "None". Absent and uncomputable both render "N/A".
    """
    value = metrics.get(key)
    return "N/A" if value is None else str(value)


def _format_performance_table(metrics: Dict[str, Any]) -> str:
    """Render performance metrics as a markdown table.

    The keys read here are the keys ValidationMetricsNode writes —
    max_drawdown_pct and win_rate_pct, not max_drawdown and win_rate. Reading
    the shorter names rendered those two rows as N/A on every run regardless of
    what was computed.
    """
    if not metrics:
        return "| Metric | Value |\n|--------|-------|\n| (no data) | N/A |"
    rows = [
        f"| Sharpe Ratio | {_cell(metrics, 'sharpe_ratio')} |",
        f"| Annualised Return | {_cell(metrics, 'annualized_return')} |",
        f"| Max Drawdown (%) | {_cell(metrics, 'max_drawdown_pct')} |",
        f"| Max Drawdown Duration (bars) | {_cell(metrics, 'max_drawdown_duration_bars')} |",
        f"| Calmar Ratio | {_cell(metrics, 'calmar_ratio')} |",
        f"| Sortino Ratio | {_cell(metrics, 'sortino_ratio')} |",
        f"| Win Rate (%) | {_cell(metrics, 'win_rate_pct')} |",
    ]
    return "| Metric | Value |\n|--------|-------|\n" + "\n".join(rows)


def _format_risk_table(metrics: Dict[str, Any]) -> str:
    """Render risk metrics as a markdown table.

    The keys read here are the keys ValidationMetricsNode writes into
    risk_metrics — var_95 / cvar_95 / annualized_volatility.
    """
    if not metrics:
        return "| Metric | Value |\n|--------|-------|\n| (no data) | N/A |"
    periods = metrics.get("drawdown_periods")
    period_count = len(periods) if isinstance(periods, list) else "N/A"
    rows = [
        f"| VaR 95% | {_cell(metrics, 'var_95')} |",
        f"| CVaR 95% | {_cell(metrics, 'cvar_95')} |",
        f"| Annualised Volatility | {_cell(metrics, 'annualized_volatility')} |",
        f"| Beta | {_cell(metrics, 'beta')} |",
        f"| Drawdown Periods | {period_count} |",
    ]
    return "| Metric | Value |\n|--------|-------|\n" + "\n".join(rows)


def _format_fsa_checks_section(fsa_checks: List[Dict[str, Any]]) -> str:
    """Render FSA check results as a markdown table."""
    if not fsa_checks:
        return (
            "| Check ID | Check Name | Result | Detail |\n"
            "|----------|-----------|--------|--------|\n"
            "| (none) | — | N/A | No FSA checks available |"
        )
    rows = []
    for check in fsa_checks:
        result = check.get("result", "N/A")
        rows.append(
            f"| {check.get('check_id', '—')} "
            f"| {check.get('check_name', '—')} "
            f"| {result} "
            f"| {check.get('detail', '—')} |"
        )
    header = "| Check ID | Check Name | Result | Detail |\n|----------|-----------|--------|--------|"
    return header + "\n" + "\n".join(rows)


def _count_fsa_results(fsa_checks: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (pass_count, fail_count) from fsa_checks list."""
    pass_count = sum(1 for c in fsa_checks if c.get("result") == "PASS")
    fail_count = sum(1 for c in fsa_checks if c.get("result") == "FAIL")
    return pass_count, fail_count


def _assemble_document(
    model_metadata: Dict[str, Any],
    validated_input: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    risk_metrics: Dict[str, Any],
    fsa_checks: List[Dict[str, Any]],
    validation_date: str,
    pass_count: int,
    fail_count: int,
    record_count: int,
) -> str:
    """Assemble the FSA-format validation document deterministically."""
    model_id: str = model_metadata.get("model_id", model_metadata.get("model_name", "unknown"))
    model_version: str = str(model_metadata.get("version", "N/A"))
    asset_class: str = model_metadata.get("asset_class", "N/A")
    strategy_type: str = model_metadata.get("strategy_type", "N/A")
    data_source: str = str(model_metadata.get("data_source", "reference"))
    # The requested window, written by the input validation step. It renders
    # verbatim into the header, so it is one of the fields that make the
    # document depend on what the caller asked for.
    validation_period: str = validated_input.get("period") or "N/A"

    overall_status = "FAIL" if fail_count > 0 else "PASS"

    # Executive summary — rendered from the numeric metrics, no model call.
    # max_drawdown_pct is the key ValidationMetricsNode writes.
    sharpe = performance_metrics.get("sharpe_ratio")
    max_dd = performance_metrics.get("max_drawdown_pct")
    exec_lines = [
        f"This document presents the FSA-compliant model validation summary for "
        f"**{model_id}** (version {model_version}), covering the period **{validation_period}**.",
        f"Asset class: **{asset_class}** | Strategy type: **{strategy_type}**.",
    ]
    if sharpe is not None:
        try:
            sharpe_f = float(sharpe)
            quality = "adequate" if sharpe_f >= 1.0 else "sub-optimal"
            exec_lines.append(
                f"The backtested Sharpe Ratio of **{sharpe}** indicates " f"{quality} risk-adjusted returns."
            )
        except (TypeError, ValueError):
            pass
    if max_dd is not None:
        exec_lines.append(f"Maximum drawdown of **{max_dd}%** was recorded during the evaluation window.")
    exec_lines.append(
        f"FSA regulatory checks: **{pass_count} PASS**, **{fail_count} FAIL** — "
        f"overall validation status: **{overall_status}**."
    )
    if fail_count > 0:
        exec_lines.append("Failed FSA checks require immediate remediation prior to deployment.")

    exec_summary = "\n\n".join(exec_lines)

    # Conclusions.
    if fail_count == 0:
        conclusion = (
            "All FSA regulatory checks passed. The model meets the required validation "
            "criteria for the specified period. No remediation actions are required at this time."
        )
    else:
        fail_names = [
            c.get("check_name", c.get("check_id", "unknown")) for c in fsa_checks if c.get("result") == "FAIL"
        ]
        conclusion = (
            f"{fail_count} FSA check(s) failed: {', '.join(fail_names)}. "
            "The model does NOT meet all regulatory validation criteria. "
            "Remediation is required before production deployment."
        )

    document = (
        f"# FSA Model Validation Document — {model_id}\n\n"
        f"**Validation Date:** {validation_date}\n"
        f"**Validation Period:** {validation_period}\n"
        f"**Model Version:** {model_version}\n"
        f"**Asset Class:** {asset_class}\n"
        f"**Strategy Type:** {strategy_type}\n"
        f"**Record Source:** {data_source}\n"
        f"**Records Analysed:** {record_count}\n"
        f"**Regulatory Framework:** FCA / FSA Model Risk Management\n\n"
        f"---\n\n"
        f"## 1. Executive Summary\n\n"
        f"{exec_summary}\n\n"
        f"---\n\n"
        f"## 2. Performance Metrics\n\n"
        f"{_format_performance_table(performance_metrics)}\n\n"
        f"---\n\n"
        f"## 3. Risk Metrics\n\n"
        f"{_format_risk_table(risk_metrics)}\n\n"
        f"---\n\n"
        f"## 4. FSA Regulatory Check Results\n\n"
        f"{_format_fsa_checks_section(fsa_checks)}\n\n"
        f"**Summary:** {pass_count} checks passed, {fail_count} checks failed.\n\n"
        f"---\n\n"
        f"## 5. Conclusions and Recommendations\n\n"
        f"{conclusion}\n\n"
        f"---\n\n"
        f"*Document generated by FIN-C2-051 Algorithmic Trading Strategy "
        f"Backtesting Summary Report agent.*\n"
    )
    return document


def _apply_output_gate(document: str, document_metadata: Dict[str, Any]) -> Optional[str]:
    """Output gate — a module-level helper, enforced inline from execute().

    It is deliberately NOT a `_extra_security_gate_output` override: the
    framework auto-wraps that hook into the graph chain, and on the clean path
    the wrapped hook returns None, so the next node receives ``state=None`` and
    the invocation raises AttributeError. Calling the check inline from
    execute() keeps every rule running without that failure mode.

    Returns a violation reason, or None when the document may be released.

    The credential scan delegates to the framework's own detector rather than
    keeping a local pattern list. That matters twice over. The framework's
    mandatory output gate scans every value of every node result with the SAME
    detector and RAISES on a match — and the wrapper then discards the node's
    whole delta, this gate's clearing included. So a local list narrower than
    the framework's is not merely incomplete, it is a bypass. And the list this
    replaced was a set of bare lowercase substrings ("secret", "password"),
    which refused any validation document that merely used one of those words.

    The reason string names the FIELD only. The matched text must never appear
    in it: the framework detector scans this node's returned values too, so an
    echoed credential would make the framework raise and discard the clearing.
    """
    if not document.strip():
        return "the generated validation document is empty"

    if detect_credentials_in_value(document):
        return "'validation_document' contains a credential-shaped value"
    if detect_credentials_in_value(document_metadata):
        return "'document_metadata' contains a credential-shaped value"
    return None


def _withheld_delta(reason: str) -> Dict[str, Any]:
    """Build the delta returned when the output gate withholds the document.

    Every output-bearing field is CLEARED, not merely omitted. LangGraph merges
    partial deltas, so a key left out of the delta keeps whatever value it
    already held in state — an omission would leave the previous node's content
    in place and the framework would project it inside the error envelope.

    `formatted_output` carries a truthy notice because the framework projects
    ``formatted_output or result``: a falsy notice re-activates the fallback
    onto `result` and produces the exact leak this function exists to prevent.
    """
    delta: Dict[str, Any] = {
        "status": AgentStatus.ERROR.value,
        "error_log": [f"DocumentGenerationNode: output withheld — {reason}"],
        "formatted_output": _WITHHELD_NOTICE,
    }
    for field in _OUTPUT_BEARING_FIELDS:
        delta[field] = {} if field == "document_metadata" else ""
    return delta


class DocumentGenerationNode(FunctionNode):
    """Assemble the model validation document from the computed metrics.

    Registered as "document_generation" in DomainWorkflowGraph (inner graph).
    Reads the computed metrics from state and produces a structured markdown
    document containing all required sections:
      1. Executive Summary
      2. Performance Metrics
      3. Risk Metrics
      4. FSA Regulatory Check Results
      5. Conclusions and Recommendations

    Generation is deterministic: every sentence is rendered from structured
    numeric state by the module-level formatters above. No model is called
    anywhere in this path, and config/agent.yaml declares
    generation_mode: deterministic to say so.

    Input state keys:
        performance_metrics: dict — sharpe_ratio, sortino_ratio, calmar_ratio,
                                    annualized_return, win_rate_pct,
                                    max_drawdown_pct, max_drawdown_duration_bars
        risk_metrics:        dict — var_95, cvar_95, annualized_volatility,
                                    beta, drawdown_periods
        fsa_checks:          list — [{check_id, check_name, result, detail}]
        model_metadata:      dict — model_id/model_name, version, asset_class,
                                    strategy_type, data_source
        validated_input:     dict — period, validation_date, backtest_records

    Output state keys (partial dict):
        validation_document: str  — the full validation document (markdown)
        llm_response:        str  — the same rendered narrative, kept for the
                                    audit trail
        result:              str  — the same document (PostProcessNode reads result)
        document_metadata:   dict — {model_id, generated_at, validation_date,
                                     fsa_check_pass_count, fsa_check_fail_count,
                                     record_count}
        formatted_output:    str  — (withheld path only) the withheld-output notice
        status:              AgentStatus.SUCCESS | AgentStatus.ERROR
        error_log:           list — (ERROR path only) error messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        validated_input: Dict[str, Any] = state.get("validated_input") or {}
        # The validation date comes from the caller's requested window. It is
        # taken from state rather than from an execute() config parameter: the
        # framework calls execute(state) with one argument, so a second
        # parameter is never populated and everything read from it would
        # silently be the default.
        validation_date: str = str(validated_input.get("validation_date") or "unknown")
        generated_at: str = validation_date

        performance_metrics: Dict[str, Any] = state.get("performance_metrics") or {}
        risk_metrics: Dict[str, Any] = state.get("risk_metrics") or {}
        fsa_checks: List[Dict[str, Any]] = state.get("fsa_checks") or []
        model_metadata: Dict[str, Any] = state.get("model_metadata") or {}
        backtest_results = state.get("backtest_results")
        record_count: int = len(backtest_results) if isinstance(backtest_results, list) else 0

        model_id: str = model_metadata.get("model_id", model_metadata.get("model_name", "unknown"))

        pass_count, fail_count = _count_fsa_results(fsa_checks)

        try:
            document: str = _assemble_document(
                model_metadata=model_metadata,
                validated_input=validated_input,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                fsa_checks=fsa_checks,
                validation_date=validation_date,
                pass_count=pass_count,
                fail_count=fail_count,
                record_count=record_count,
            )
        except Exception:  # noqa: BLE001
            # Summarised on purpose: the exception text can carry document
            # content and source paths, and error_log is an operator-visible
            # field.
            logger.exception("DocumentGenerationNode: document assembly failed")
            return _withheld_delta("the validation document could not be assembled")

        document_metadata: Dict[str, Any] = {
            "model_id": model_id,
            "generated_at": generated_at,
            "validation_date": validation_date,
            "fsa_check_pass_count": pass_count,
            "fsa_check_fail_count": fail_count,
            "record_count": record_count,
        }

        # Output gate — enforced inline before anything is returned.
        violation = _apply_output_gate(document, document_metadata)
        if violation:
            logger.error("DocumentGenerationNode: output withheld — %s", violation)
            return _withheld_delta(violation)

        logger.info(
            "DocumentGenerationNode: document assembled — " "model=%s fsa_pass=%d fsa_fail=%d records=%d chars=%d",
            model_id,
            pass_count,
            fail_count,
            record_count,
            len(document),
        )

        # Audit trail: record the document-generation side-effect.
        # Free-function form only — the method form raises AttributeError.
        emit_trace_event(
            "document_generation_complete",
            {
                "model_id": model_id,
                "fsa_pass": pass_count,
                "fsa_fail": fail_count,
                "record_count": record_count,
                "doc_length_chars": len(document),
            },
            state,
        )

        released = {
            "validation_document": document,
            "llm_response": document,
            "result": document,
            "document_metadata": document_metadata,
            "status": AgentStatus.SUCCESS.value,
        }
        # Inventory guard: every output-bearing field this node returns must be
        # one the gate knows how to clear. A new field added above without being
        # listed in _OUTPUT_BEARING_FIELDS would otherwise survive a withheld
        # run untouched and be projected inside the error envelope.
        unlisted = set(released) - set(_OUTPUT_BEARING_FIELDS) - {"status"}
        if unlisted:
            raise AssertionError(
                "DocumentGenerationNode returns output fields the gate does not "
                f"clear: {sorted(unlisted)}. Add them to _OUTPUT_BEARING_FIELDS."
            )
        return released
