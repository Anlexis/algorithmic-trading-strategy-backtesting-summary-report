"""AgentCore Platform v1.0"""

# FIN-C2-051 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full algorithmic trading backtest validation domain
# workflow:
#
#   START → backtest_ingestion → validation_metrics → document_generation → END
#
# Called by FinC2051GraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   Inherits BaseGraph (fully custom topology — no forced backbone)
#   Implements all 7 BaseGraph ABC methods
#   register_nodes() does NOT call super() (abstract in BaseGraph)
#   Does NOT register initialize / finalize (outer backbone concerns)
#   get_output() designed together with FinC2051GraphNode.merge_output()
#   _extra_initial_state() seeds the bridged payload (src/graph/context_bridge.py)
#   No platform-internal imports

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_payload
from src.nodes.backtest_ingestion_node import BacktestIngestionNode
from src.nodes.validation_metrics_node import ValidationMetricsNode
from src.nodes.document_generation_node import DocumentGenerationNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for FIN-C2-051.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by FinC2051GraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          → backtest_ingestion   (BacktestIngestionNode)
          → validation_metrics   (ValidationMetricsNode)
          → document_generation  (DocumentGenerationNode)
          → END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns — not registered here.

    Error strategy:
        BacktestIngestionNode returning empty records → continue
        (DocumentGenerationNode generates an N/A document, not an error).
        Unhandled exceptions bubble up to FinC2051GraphNode.error_strategy
        in the outer graph.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "fin_c2_051_backtest_validation_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The only accepted key is ``fsa_thresholds``: a mapping of threshold
        name to number, already shape- and range-checked by the outer graph
        before it is forwarded here. Absence is non-fatal — ValidationMetricsNode
        falls back to its documented defaults. A non-mapping value is dropped
        for the same reason rather than failing graph construction.
        """
        config: Dict[str, Any] = dict(getattr(self, "config", None) or {})
        if not isinstance(config.get("fsa_thresholds", {}), dict):
            config.pop("fsa_thresholds", None)
            self.config = config

    # ── Initial-state seeding ─────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the validated payload the outer graph stashed on the bridge.

        The framework invokes this graph as ``invoke(user_input, ...)``, which
        places whatever FinC2051GraphNode.extract_input() returned on
        ``user_input`` and nothing else. The domain nodes read
        ``validated_input``, so without this hook they see an absent payload on
        every real invocation and refuse the request. The bridge
        (src/graph/context_bridge.py) carries it across; this hook seeds it.

        Returns an empty mapping when nothing was stashed — a direct
        ``DomainWorkflowGraph().invoke()`` (unit tests, internal callers) then
        behaves exactly as before.
        """
        payload = get_caller_payload()
        return {"validated_input": payload} if payload else {}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 3 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        Every key registered here is referenced in add_edges().
        """
        config: Dict[str, Any] = dict(getattr(self, "config", None) or {})
        thresholds = config.get("fsa_thresholds") or {}
        self._nodes["backtest_ingestion"] = BacktestIngestionNode()
        self._nodes["validation_metrics"] = ValidationMetricsNode(thresholds=thresholds)
        self._nodes["document_generation"] = DocumentGenerationNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear backtest validation domain topology.

        Each step passes its partial-dict output into the shared State.
        The topology is intentionally linear — no conditional branching
        between domain nodes. route() is implemented as required by the ABC
        but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "backtest_ingestion")
        self._sg.add_edge("backtest_ingestion", "validation_metrics")
        self._sg.add_edge("validation_metrics", "document_generation")
        self._sg.add_edge("document_generation", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by BaseGraph ABC.

        Linear-topology invariant: the pipeline is wired entirely with
        unconditional add_edge() calls in add_edges() and
        add_conditional_edges() is never used, so route() is never
        invoked at runtime. It is implemented only to satisfy the BaseGraph
        ABC.

        Both branches return END so that an unexpected call can never
        re-enter a mid-graph node (returning a node name like
        "document_generation" would risk an infinite loop). The error branch
        is kept explicit for readability.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return END

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by FinC2051GraphNode.merge_output() in graph.py
        as the sub_result argument. Both methods are designed together to
        guarantee field-name consistency:

            Inner get_output()   emits: "validation_document", "result",
                                        "document_metadata", "llm_response",
                                        "status", "validation_metadata",
                                        "trace_id", ...
            Outer merge_output() reads: sub_result.get("result"),
                                        sub_result.get("validation_document"),
                                        sub_result.get("document_metadata"),
                                        sub_result.get("llm_response")

        document_metadata and llm_response are produced by DocumentGenerationNode
        and MUST be surfaced here — merge_output() reads them off sub_result and
        maps them into the outer state delta. Omitting them leaves them None
        end-to-end (PostProcessNode's output gate then withholds the response
        because document_metadata is empty).

        Additional fields (trace_id, correlation_id, node_history) are
        surfaced for observability and downstream extension.

        error_log is surfaced deliberately: the framework wraps a non-success
        sub_result in a SubgraphError built from ``sub_result["error_log"]``.
        Omitting the key made every inner failure surface to the operator as
        "failed: []" with no reason attached.

        formatted_output carries the withheld-output notice raised by the
        document gate. The framework projects ``formatted_output or result``,
        so a truthy notice is what keeps a blocked run from falling back to
        whatever ``result`` still holds.
        """
        return {
            "validation_document": state.get("validation_document"),
            "result": state.get("result"),
            "document_metadata": state.get("document_metadata", {}),
            "llm_response": state.get("llm_response"),
            "validation_metadata": state.get("validation_metadata", {}),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
            "error_log": state.get("error_log", []),
            "formatted_output": state.get("formatted_output"),
        }
