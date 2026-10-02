# FIN-C2-051 — Unit Tests: Graph (outer AgentBaseGraph + inner BaseGraph)
#
# Tests:
#   - FinC2051Agent instantiates (outer graph compiles), name + state_schema
#   - FinC2051GraphNode.extract_input() round-trip (validated_input dict + JSON fallback)
#   - FinC2051GraphNode.merge_output() maps sub_result → result (changed keys only)
#   - FinC2051GraphNode.get_subgraph() returns a DomainWorkflowGraph
#   - DomainWorkflowGraph instantiates with all BaseGraph ABC methods implemented
#
# Framework-dependent instantiation is wrapped in ImportError skips, matching
# the peer template idiom (the SDK wheel is not present in a bare local checkout).

import json

import pytest


class TestFinC2051AgentInstantiation:
    """FinC2051Agent (outer graph) must instantiate and expose identity props."""

    def test_agent_instantiates(self):
        """FinC2051Agent() must not raise at import/instantiation."""
        try:
            from src.graph.graph import FinC2051Agent

            agent = FinC2051Agent()
            assert agent is not None
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

    def test_graph_alias_is_agent(self):
        """The module-level Graph alias must point at FinC2051Agent."""
        try:
            from src.graph.graph import Graph, FinC2051Agent

            assert Graph is FinC2051Agent
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

    def test_agent_name_property(self):
        """FinC2051Agent.name must return 'FIN-C2-051'."""
        try:
            from src.graph.graph import FinC2051Agent

            agent = FinC2051Agent()
            assert agent.name == "FIN-C2-051", f"Expected name='FIN-C2-051', got {agent.name!r}"
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

    def test_state_schema_is_state(self):
        """FinC2051Agent.state_schema must return the State class."""
        try:
            from src.graph.graph import FinC2051Agent
            from src.schemas.state import State

            agent = FinC2051Agent()
            assert agent.state_schema is State
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")


class TestFinC2051GraphNodeMethods:
    """FinC2051GraphNode.extract_input() / merge_output() / get_subgraph()."""

    def test_extract_input_from_validated_input_dict(self):
        """extract_input returns the validated_input dict when present."""
        try:
            from src.graph.graph import FinC2051GraphNode
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        node = FinC2051GraphNode()
        validated = {
            "model_id": "MODEL-ALPHA",
            "start_date": "2024-01-01",
            "end_date": "2025-12-01",
            "asset_class": "equity",
        }
        state = {"validated_input": validated, "user_input": "should not be used"}

        extracted = node.extract_input(state)
        assert extracted == validated, f"extract_input should return validated_input dict, got {extracted!r}"

    def test_extract_input_requires_validated_input(self):
        """extract_input reads validated_input only — never the raw user_input.

        Parsing the raw string here would hand the inner graph a payload that
        the validation step never checked, so an unvalidated request would
        reach the domain nodes whenever validated_input happened to be absent.
        """
        try:
            from src.graph.graph import FinC2051GraphNode
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        node = FinC2051GraphNode()
        payload = {"model_id": "MODEL-BETA", "asset_class": "fx"}

        assert node.extract_input({"user_input": json.dumps(payload)}) == {}
        assert node.extract_input({"validated_input": payload}) == payload

    def test_extract_input_stashes_payload_on_the_bridge(self):
        """extract_input publishes the payload the inner graph seeds from.

        The framework passes extract_input()'s return value to the inner graph
        as user_input and forwards nothing else, so the bridge is the only way
        validated_input reaches the domain nodes.
        """
        try:
            from src.graph.context_bridge import get_caller_payload
            from src.graph.graph import FinC2051GraphNode
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        payload = {"model_id": "MODEL-BETA", "asset_class": "fx"}
        FinC2051GraphNode().extract_input({"validated_input": payload})
        assert get_caller_payload() == payload

    def test_extract_input_empty_state_returns_empty_dict(self):
        """extract_input on an empty state returns an empty dict (no crash)."""
        try:
            from src.graph.graph import FinC2051GraphNode
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        node = FinC2051GraphNode()
        extracted = node.extract_input({})
        assert extracted == {}, f"Expected empty dict, got {extracted!r}"

    def test_merge_output_maps_result(self):
        """merge_output must map sub_result.result and validation_document into the delta."""
        try:
            from src.graph.graph import FinC2051GraphNode
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        node = FinC2051GraphNode()
        sub_result = {
            "result": "# FSA Validation Document\n\nContent.",
            "validation_document": "# FSA Validation Document\n\nContent.",
            "document_metadata": {"model_id": "MODEL-ALPHA", "fsa_check_fail_count": 0},
            "llm_response": "# FSA Validation Document\n\nContent.",
            "status": "success",
        }

        merged = node.merge_output({}, sub_result)
        assert (
            merged.get("result") == "# FSA Validation Document\n\nContent."
        ), "merge_output must map result from sub_result"
        assert merged.get("validation_document") == "# FSA Validation Document\n\nContent."
        assert merged.get("document_metadata", {}).get("model_id") == "MODEL-ALPHA"

    def test_merge_output_returns_only_changed_keys(self):
        """merge_output must NOT echo the full outer state back."""
        try:
            from src.graph.graph import FinC2051GraphNode
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        node = FinC2051GraphNode()
        outer_state = {
            "user_input": "original",
            "validated_input": {"model_id": "MODEL-ALPHA"},
            "session_id": "sess-001",
        }
        sub_result = {
            "result": "# Report",
            "validation_document": "# Report",
            "document_metadata": {},
            "llm_response": "# Report",
        }

        merged = node.merge_output(outer_state, sub_result)
        assert "user_input" not in merged, "merge_output must NOT return outer state fields"
        assert "session_id" not in merged, "merge_output must NOT include session_id"
        assert "result" in merged
        assert "validation_document" in merged

    def test_get_subgraph_returns_domain_workflow_graph(self):
        """get_subgraph() must return a DomainWorkflowGraph instance."""
        try:
            from src.graph.graph import FinC2051GraphNode
            from src.graph.domain_workflow_graph import DomainWorkflowGraph
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        node = FinC2051GraphNode()
        subgraph = node.get_subgraph()
        assert isinstance(
            subgraph, DomainWorkflowGraph
        ), f"get_subgraph() must return DomainWorkflowGraph, got {type(subgraph)}"


class TestDomainWorkflowGraph:
    """DomainWorkflowGraph (inner BaseGraph) must implement all ABC methods."""

    def test_domain_workflow_graph_instantiates(self):
        """DomainWorkflowGraph() must not raise NotImplementedError at instantiation."""
        try:
            from src.graph.domain_workflow_graph import DomainWorkflowGraph

            graph = DomainWorkflowGraph()
            assert graph is not None
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")
        except NotImplementedError as exc:
            pytest.fail(f"DomainWorkflowGraph has unimplemented ABC methods: {exc}")

    def test_domain_workflow_graph_name(self):
        """DomainWorkflowGraph.name must be the FIN-C2-051 inner workflow id."""
        try:
            from src.graph.domain_workflow_graph import DomainWorkflowGraph

            graph = DomainWorkflowGraph()
            assert (
                graph.name == "fin_c2_051_backtest_validation_workflow"
            ), f"Unexpected inner graph name: {graph.name!r}"
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

    def test_domain_workflow_graph_state_schema(self):
        """DomainWorkflowGraph.state_schema must be the shared State class."""
        try:
            from src.graph.domain_workflow_graph import DomainWorkflowGraph
            from src.schemas.state import State

            graph = DomainWorkflowGraph()
            assert graph.state_schema is State
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

    def test_get_output_shape(self):
        """get_output() must surface validation_document, result, and status keys."""
        try:
            from src.graph.domain_workflow_graph import DomainWorkflowGraph
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        graph = DomainWorkflowGraph()
        state = {
            "validation_document": "# Doc",
            "result": "# Doc",
            "status": "success",
        }
        out = graph.get_output(state)
        assert out.get("validation_document") == "# Doc"
        assert out.get("result") == "# Doc"
        assert "status" in out

    def test_get_output_surfaces_document_metadata_and_llm_response(self):
        """F-1: get_output() must emit document_metadata and llm_response so
        merge_output() (which reads them off sub_result) does not get None."""
        try:
            from src.graph.domain_workflow_graph import DomainWorkflowGraph
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        graph = DomainWorkflowGraph()
        state = {
            "validation_document": "# Doc",
            "result": "# Doc",
            "llm_response": "# Doc",
            "document_metadata": {"model_id": "MODEL-ALPHA", "fsa_check_fail_count": 0},
            "status": "success",
        }
        out = graph.get_output(state)
        assert (
            out.get("document_metadata", {}).get("model_id") == "MODEL-ALPHA"
        ), "get_output() must surface document_metadata for merge_output()"
        assert out.get("llm_response") == "# Doc", "get_output() must surface llm_response for merge_output()"

    def test_get_output_feeds_merge_output_end_to_end(self):
        """F-1 regression: inner get_output() → outer merge_output() must carry
        document_metadata and llm_response through (not None end-to-end)."""
        try:
            from src.graph.graph import FinC2051GraphNode
            from src.graph.domain_workflow_graph import DomainWorkflowGraph
        except ImportError as exc:
            pytest.skip(f"Framework not installed in CI: {exc}")

        inner_state = {
            "validation_document": "# Doc",
            "result": "# Doc",
            "llm_response": "# Doc",
            "document_metadata": {"model_id": "MODEL-ALPHA", "fsa_check_fail_count": 0},
            "status": "success",
        }
        sub_result = DomainWorkflowGraph().get_output(inner_state)
        merged = FinC2051GraphNode().merge_output({}, sub_result)

        assert (
            merged.get("document_metadata", {}).get("model_id") == "MODEL-ALPHA"
        ), "document_metadata must survive get_output()→merge_output()"
        assert merged.get("llm_response") == "# Doc", "llm_response must survive get_output()→merge_output()"
