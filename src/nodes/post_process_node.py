"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from api/ or from another agent
#
# FIN-C2-051 — PostProcessNode
# Outer backbone post_process slot — the response output gate.
# Reads state["result"] (the validation document produced by the main node /
# inner subgraph) and applies the output content gate before the response is
# returned to the caller.
#
# Rules:
#   - Verify result is non-empty
#   - Verify document_metadata is populated
#   - Scan for credential shapes using the framework's own detector, plus the
#     assignment forms below that the framework detector does not cover
#   - On a violation: return ERROR, CLEAR every output-bearing field, and set a
#     truthy withheld-output notice
#   - required_trust_level = TrustLevel.VERIFIED_EXTERNAL

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security.credential_detector import detect_credentials_in_value

# Audit trail — the free-function form; the instance-method form does not exist.
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disallowed-content patterns
#
# The credential scan delegates to the framework's own detector, which covers
# provider key prefixes, JWTs, AWS access key IDs, Bearer tokens and database
# connection strings. Keeping a LOCAL list instead would be a bypass rather
# than a second opinion: the framework's mandatory output gate runs the same
# detector over every value this node returns and RAISES on a match, and the
# wrapper then discards the node's whole delta — this gate's clearing included.
# So the local set may only ever ADD to the framework's, never narrow it.
#
# The one form the framework detector does not carry is the assignment shape
# (`password = ...`, `token: ...`), which is what a leaked configuration dump
# looks like. That stays here as an addition.
# ---------------------------------------------------------------------------
_EXTRA_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

# Output-bearing fields cleared when the gate withholds the response. The
# inventory is explicit so a field added to the state schema later cannot
# quietly stay populated inside an error envelope.
_OUTPUT_BEARING_FIELDS: Tuple[str, ...] = (
    "result",
    "validation_document",
    "llm_response",
    "document_metadata",
)

# The notice that replaces the response when the gate withholds it. It must be
# TRUTHY: the framework projects `formatted_output or result`, so an empty
# notice would re-activate the fallback onto whatever `result` still holds and
# ship the ungated document inside the error envelope.
_WITHHELD_NOTICE = (
    "The validation document was withheld: the output gate found content that "
    "must not leave the agent. No document is available for this request."
)


def _apply_output_gate(content: str) -> Optional[str]:
    """Run the output content gate.

    Returns the NAME of the first matched pattern class, or None if clean.
    The matched text is never returned: the framework's mandatory gate scans
    every value this node returns, so echoing a credential into the reason
    would make the framework raise and discard the clearing.
    """
    findings = detect_credentials_in_value(content)
    if findings:
        return str(findings[0]["type"])
    for name, pattern in _EXTRA_PATTERNS:
        if pattern.search(content):
            return name
    return None


def _withheld_delta(reason: str) -> Dict[str, Any]:
    """Build the delta returned when the gate withholds the response.

    Every output-bearing field is CLEARED, not merely omitted. LangGraph merges
    partial deltas, so a key left out of the delta keeps whatever value it
    already held in state — the ungated document would survive and be projected
    inside the error envelope.
    """
    delta: Dict[str, Any] = {
        "status": AgentStatus.ERROR.value,
        "error_log": [f"PostProcessNode: output withheld — {reason}"],
        "formatted_output": _WITHHELD_NOTICE,
    }
    for field in _OUTPUT_BEARING_FIELDS:
        delta[field] = {} if field == "document_metadata" else ""
    return delta


class PostProcessNode(FunctionNode):
    """Response output gate: scan the validation document for disallowed content.

    Outer backbone post_process slot. Reads state["result"] (the validation
    report produced by the main node) and applies the output content gate
    before the response is returned to the caller.

    Input state keys:
        result:            final validation report (from main / inner graph)
        document_metadata: metadata dict (must be populated; checked by the gate)

    Output state keys (partial dict — only what this node changes):
        result:              the released report, or "" when withheld
        validation_document: cleared when withheld
        llm_response:        cleared when withheld
        document_metadata:   cleared when withheld
        formatted_output:    the withheld-output notice, when withheld
        status:              AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:           (on error) list of error messages
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    #
    # The output checks (non-empty result, populated document_metadata,
    # disallowed-content scan) run INLINE in execute(). They are deliberately
    # NOT a `_extra_security_gate_output` override: the framework auto-wraps
    # that hook into the graph chain, and on the clean path the wrapped hook
    # returns None, so the next node receives state=None and the invocation
    # raises AttributeError. Calling the checks inline keeps every rule running
    # without that failure mode.

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result: str = state.get("result") or ""
        document_metadata: Any = state.get("document_metadata") or {}
        # model_id lives in document_metadata / model_metadata (no top-level
        # state["model_id"] is written by upstream nodes); the bare
        # state["model_id"] fallback covers direct callers.
        model_id: str = (
            (document_metadata.get("model_id") if isinstance(document_metadata, dict) else "")
            or (state.get("model_metadata") or {}).get("model_id")
            or state.get("model_id")
            or ""
        )

        # Number of failed regulatory checks, carried through for the audit
        # record (0 when the upstream node did not set it).
        fsa_check_fail_count: int = state.get("fsa_check_fail_count") or 0

        # --- Verify result is non-empty ---
        # An empty result is reported as an ERROR with a truthy notice, not as
        # a success carrying no output. A success envelope whose output field
        # is empty gives the caller no signal that anything went wrong, which
        # is worse than a refusal it can act on.
        if not result.strip():
            logger.warning("PostProcessNode: no validation document was produced")
            return _withheld_delta("no validation document was produced")

        # --- Verify document_metadata is populated ---
        if not document_metadata:
            logger.error("PostProcessNode: document_metadata is absent — the output cannot be validated")
            return _withheld_delta("'document_metadata' is missing")

        # --- Content scan over the released field ---
        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=result,
            domain="FIN AlgorithmicTradingBacktestingAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(result, str) and not _apply_output_gate(result + _review):
            result = result + _review

        violation = _apply_output_gate(result)
        if violation:
            logger.error("PostProcessNode: output withheld — pattern class %s", violation)
            return _withheld_delta(f"'result' matched the {violation} pattern class")

        logger.info(
            "PostProcessNode: output gate passed — %d chars, model_id=%s",
            len(result),
            model_id,
        )

        # Audit trail: record the post_process side-effect.
        # Free-function form only — the method form raises AttributeError.
        emit_trace_event(
            "post_process_complete",
            {
                "model_id": model_id,
                "fsa_check_fail_count": fsa_check_fail_count,
                "result_length": len(result),
                "document_metadata_keys": (
                    sorted(document_metadata.keys()) if isinstance(document_metadata, dict) else []
                ),
            },
            state,
        )

        return {
            "result": result,
            "status": AgentStatus.SUCCESS.value,
        }
