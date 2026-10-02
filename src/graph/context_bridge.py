"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the validated invocation payload across
# the outer→inner graph boundary.
#
# Why this exists: the framework invokes an inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)`. Whatever
# `extract_input()` returns therefore lands on the inner graph's `user_input`
# key and NOWHERE else — the outer state's `validated_input` and
# `input_context` are not forwarded. The inner domain nodes read
# `validated_input`, so without a bridge they see an absent payload on every
# real invocation and refuse the request.
#
# The payload crosses the boundary out-of-band, through the two sanctioned
# subclass hooks:
#
#   FinC2051GraphNode.extract_input(state)     [runs BEFORE subgraph.invoke]
#       → set_caller_payload(<validated payload>)
#   DomainWorkflowGraph._extra_initial_state() [runs INSIDE subgraph.invoke]
#       → returns {"validated_input": get_caller_payload()}
#
# A ContextVar keeps the hand-off correct per thread/task, so concurrent
# invocations in one process cannot observe each other's payload.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_PAYLOAD: ContextVar[Optional[Dict[str, Any]]] = ContextVar("fin_c2_051_caller_payload", default=None)


def set_caller_payload(payload: Optional[Dict[str, Any]]) -> None:
    """Stash the validated payload for the imminent inner-graph invoke."""
    _CALLER_PAYLOAD.set(dict(payload) if payload else {})


def get_caller_payload() -> Dict[str, Any]:
    """Read (without consuming) the stashed payload; {} when none was set."""
    return _CALLER_PAYLOAD.get() or {}
