# FIN-C2-051 — Proof of boundary: the declared trust level can actually run the graph
#
# The framework's trust gate runs inside the node wrapper, BEFORE execute() is
# called, and it is ordered ANONYMOUS < VERIFIED_EXTERNAL < INTERNAL. A node
# that demands a level higher than the manifest's declared entry level can
# never be reached by the caller the manifest describes — and because the unit
# suite calls execute() directly, the wrapper never runs there and the whole
# class of defect stays invisible.
#
# So the lattice is asserted two ways: statically, that no node outranks the
# manifest, and dynamically, that a caller holding exactly the declared level
# gets an answer through the real entry point.

import json
import pathlib

import pytest

_MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "config" / "agent.yaml"

_REQUEST = {
    "model_id": "MODEL-ALPHA",
    "asset_class": "equity",
    "backtest_data": "inline:reference",
    "validation_date_start": "2022-01-01",
    "validation_date_end": "2024-12-01",
}


def _declared_trust_level():
    import yaml

    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    return manifest["required_trust_level"]


def _domain_nodes():
    from src.graph.domain_workflow_graph import DomainWorkflowGraph
    from src.nodes.post_process_node import PostProcessNode
    from src.nodes.pre_process_node import PreProcessNode

    graph = DomainWorkflowGraph()
    graph.register_nodes()
    nodes = dict(graph._nodes)
    nodes["pre_process"] = PreProcessNode()
    nodes["post_process"] = PostProcessNode()
    return nodes


@pytest.fixture(autouse=True)
def _require_framework():
    try:
        import src.graph.graph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - framework absent locally
        pytest.skip(f"Framework not installed: {exc}")


class TestNoNodeOutranksTheManifest:
    def test_every_node_is_reachable_at_the_declared_level(self):
        from framework.schemas.trust_level import TrustLevel

        order = [
            TrustLevel.ANONYMOUS,
            TrustLevel.VERIFIED_EXTERNAL,
            TrustLevel.INTERNAL,
        ]
        declared = TrustLevel(_declared_trust_level())
        for name, node in _domain_nodes().items():
            required = type(node).required_trust_level
            assert order.index(required) <= order.index(declared), (
                f"{name} requires {required.value}, which a caller holding the "
                f"manifest's declared {declared.value} can never present"
            )

    def test_every_node_declares_its_level_explicitly(self):
        for name, node in _domain_nodes().items():
            assert (
                "required_trust_level" in type(node).__dict__
            ), f"{name} inherits its trust level instead of declaring one"


class TestTheDeclaredCallerGetsAnAnswer:
    def test_an_invocation_at_the_declared_level_succeeds(self):
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel
        from framework.secrets.context import bound_secrets

        import src.api.server as server

        agent = server.agent
        with bound_secrets(agent._secrets_provider):
            context = InvocationContext(
                session_id="pb-trust",
                caller_trust_level=TrustLevel(_declared_trust_level()),
            )
            result = agent.invoke(json.dumps(_REQUEST), ctx=context)

        assert result["status"] == "success", result
        assert result["output"], "the declared caller received no output"

    def test_an_anonymous_invocation_is_denied_without_leaking_content(self):
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel
        from framework.secrets.context import bound_secrets

        import src.api.server as server

        agent = server.agent
        with bound_secrets(agent._secrets_provider):
            context = InvocationContext(
                session_id="pb-anon",
                caller_trust_level=TrustLevel.ANONYMOUS,
            )
            result = agent.invoke(json.dumps(_REQUEST), ctx=context)

        assert result["status"] == "error", result
        assert not result["output"]
