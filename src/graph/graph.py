"""AgentCore Platform v1.0"""

# FIN-C2-051 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — identical to Cat 1, do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max_retry)
#                                          pre_process
#
#   `main` slot is a GraphNode subclass (FinC2051GraphNode) that delegates
#   the full domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (multi-step topology)
#   src/graph/context_bridge.py        ← validated-payload hand-off (outer → inner)
#
# Rules enforced:
#   FinC2051Agent inherits AgentBaseGraph
#   super().register_nodes() called first (fills initialize + finalize)
#   FinC2051GraphNode assigned to self._nodes["main"]
#   merge_output() returns only changed keys
#   add_edges() NOT overridden on the outer graph
#   No platform-internal imports

import math
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

import yaml

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_state import AgentState
from src.graph.context_bridge import set_caller_payload
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

_RUNTIME_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Accepted ranges for the operator-declared FSA thresholds. A declared value
# outside its range (or of the wrong shape) is not forwarded — the pipeline then
# runs on the documented built-in default rather than on a nonsense threshold.
_FSA_THRESHOLD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "fsa_chk01_min_sharpe": (-100.0, 100.0),
    "fsa_chk02_max_drawdown_pct": (0.0, 100.0),
    "fsa_chk03_min_backtest_months": (0.0, 600.0),
    "fsa_chk04_min_oos_months": (0.0, 600.0),
}


def runtime_config() -> Dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and passes as
    ``Graph(config=...)``; the standalone entry point (src/api/server.py) reads
    it here so a registry-loaded agent and a standalone one see identical
    configuration. Returns an empty dict — never raises — when the file is
    absent, unreadable, not valid YAML, or not a mapping; the graph then runs on
    its built-in defaults.
    """
    try:
        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a broken config file must not break startup
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _finite_in_range(value: Any, low: float, high: float) -> Optional[float]:
    """Validate a declared numeric setting: a real number, finite, within bounds.

    Bools, strings, non-numerics, NaN/±Infinity and out-of-range values return
    None. The non-finite case is the dangerous one: NaN compares False against
    everything, so an unvalidated NaN threshold would silently disable the check
    it feeds instead of failing it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if not low <= number <= high:
        return None
    return number


def _domain_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the validated domain settings forwarded to the inner graph.

    Only keys that pass their shape and range check are forwarded; anything else
    is dropped so the node keeps its documented default.
    """
    declared = config.get("fsa_thresholds")
    if not isinstance(declared, dict):
        return {}
    thresholds: Dict[str, float] = {}
    for key, (low, high) in _FSA_THRESHOLD_BOUNDS.items():
        parsed = _finite_in_range(declared.get(key), low, high)
        if parsed is not None:
            thresholds[key] = parsed
    return {"fsa_thresholds": thresholds} if thresholds else {}


class FinC2051GraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of FinC2051Agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph).
    Called by the backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    — instantiate and return DomainWorkflowGraph
      extract_input()   — pull the validated payload from outer state
      merge_output()    — map sub_result fields into outer state delta (changed keys only)
      error_strategy    — "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # S-1 declared on the wrapper too: the CI gate only AST-scans FunctionNode
    # subclasses, so a GraphNode main slot passes the pipeline without one and is
    # flagged at review. Same level the nodes in this repo already declare.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    # "propagate": re-raise inner graph exceptions as SubgraphError (default — fail fast).
    # "handle": call on_subgraph_error() instead — use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only.
    # True: surface inner HITL interrupt to the outer caller.
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """Hold the validated domain settings forwarded to the inner graph."""
        super().__init__()
        self._settings: Dict[str, Any] = settings or {}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._settings)

    def extract_input(self, state: AgentState) -> Dict[str, Any]:
        """Return the input passed into inner_graph.invoke(), and bridge it.

        PreProcessNode validates the raw user_input and the structured caller
        parameters, and writes the result to validated_input.

        The returned value lands on the inner graph's `user_input` key only —
        the inner domain nodes read `validated_input`. set_caller_payload()
        therefore stashes the same payload for the inner graph's
        _extra_initial_state() hook to seed (src/graph/context_bridge.py).
        """
        validated = state.get("validated_input")
        payload: Dict[str, Any]
        if isinstance(validated, dict) and validated:
            payload = validated
        else:
            payload = {}
        set_caller_payload(payload)
        return payload

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  → "result", "validation_document",
                                      "document_metadata", "llm_response", ...
          This merge_output() reads → sub_result.get("result"),
                                      sub_result.get("validation_document"),
                                      sub_result.get("document_metadata"),
                                      sub_result.get("llm_response")

        result (str | None): primary backtesting report output; PostProcessNode
          reads state["result"] for the output gate.
        validation_document (str | None): generated validation document.
        document_metadata (dict | None): metadata for the generated document.
        llm_response (str | None): generated narrative retained for the audit trail.
        formatted_output (str | None): withheld-output notice raised by the inner
          gate; carried through because the framework projects it ahead of result.
        """
        merged: Dict[str, Any] = {
            "result": sub_result.get("result"),
            "validation_document": sub_result.get("validation_document", ""),
            "document_metadata": sub_result.get("document_metadata", {}),
            "llm_response": sub_result.get("llm_response", ""),
        }
        notice = sub_result.get("formatted_output")
        if notice:
            merged["formatted_output"] = notice
        return merged


class FinC2051Agent(AgentBaseGraph):
    """Outer graph for FIN-C2-051 (Cat 2).

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    FinC2051GraphNode (main slot), which delegates to DomainWorkflowGraph
    (inner BaseGraph).

    Backbone (fixed — identical to Cat 1):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode (input validation)
      - main:        FinC2051GraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "FIN-C2-051"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).

        The declared runtime settings are validated here and forwarded to the
        inner graph, so a value written in config/config.yaml reaches the node
        that reads it instead of being silently ignored.
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = FinC2051GraphNode(settings=_domain_settings(self.config))
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Module-level alias so callers can import `from src.graph.graph import Graph`.
Graph = FinC2051Agent
