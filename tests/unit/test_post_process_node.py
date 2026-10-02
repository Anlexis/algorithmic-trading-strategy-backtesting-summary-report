# FIN-C2-051 — Unit Tests: PostProcessNode (TC-07, TC-08, TC-10)
#
# TC-07: the output gate blocks an API key pattern in result
# TC-08: required_trust_level = TrustLevel.VERIFIED_EXTERNAL
# TC-10: the output gate blocks JWT / Bearer-token patterns in result
#
# PostProcessNode.execute(state) reads result + document_metadata + model_id.
# A clean result requires a populated document_metadata to pass the gate.
#
# Withholding assertions check PRESENCE **and** value. LangGraph merges partial
# deltas, so a key the node omits keeps whatever it already held in state — an
# `assert not result.get(field)` would therefore pass on a gate that clears
# nothing at all, which is the exact defect these tests exist to catch.


def _status_ok(result) -> bool:
    from framework.schemas.agent_status import AgentStatus

    return result.get("status") in (AgentStatus.SUCCESS, AgentStatus.SUCCESS.value)


def _status_err(result) -> bool:
    from framework.schemas.agent_status import AgentStatus

    return result.get("status") in (AgentStatus.ERROR, AgentStatus.ERROR.value)


def _assert_withheld(result, label=""):
    """Assert the delta withholds output completely.

    Every output-bearing key must be PRESENT in the returned delta and empty,
    and formatted_output must be TRUTHY — the framework projects
    `formatted_output or result`, so a falsy notice re-activates the fallback
    onto result and ships the ungated document inside the error envelope.
    """
    from src.nodes.post_process_node import _OUTPUT_BEARING_FIELDS

    assert _status_err(result), f"{label}: expected ERROR, got {result.get('status')}"
    for field in _OUTPUT_BEARING_FIELDS:
        assert field in result, (
            f"{label}: {field!r} missing from the delta — an omitted key keeps its "
            "previous value once LangGraph merges the partial update"
        )
        assert not result[field], f"{label}: {field!r} was not cleared: {result[field]!r}"
    notice = result.get("formatted_output")
    assert notice, f"{label}: formatted_output must be truthy, got {notice!r}"
    assert isinstance(notice, str) and notice.strip()


def _metadata():
    return {
        "model_id": "MODEL-ALPHA",
        "generated_at": "2026-01-01",
        "validation_date": "2026-01-01",
        "fsa_check_pass_count": 4,
        "fsa_check_fail_count": 1,
    }


class TestTC07S3BlocksApiKey:
    """TC-07: the output gate blocks an API key pattern in result."""

    def test_api_key_in_result_is_blocked(self):
        """result containing an API key pattern must return ERROR + sanitized stub."""
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        result = node.execute(
            {
                "result": "FSA validation document... sk-1234567890abcdefghij1234567890 ...end",
                "document_metadata": _metadata(),
                "model_id": "MODEL-ALPHA",
            }
        )

        _assert_withheld(result, "api key")

    def test_credential_assignment_in_result_is_blocked(self):
        """result containing a credential assignment (password=...) must return ERROR."""
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        result = node.execute(
            {
                "result": "Config dump leaked: password = hunter2supersecret value",
                "document_metadata": _metadata(),
                "model_id": "MODEL-ALPHA",
            }
        )

        _assert_withheld(result, "credential assignment")

    def test_clean_result_passes_gate(self):
        """Clean result with populated metadata must pass the output gate."""
        from src.nodes.post_process_node import PostProcessNode

        clean_report = (
            "# FSA Model Validation Document\n\n"
            "All FSA regulatory checks passed. The model meets the required criteria."
        )
        node = PostProcessNode()
        result = node.execute(
            {
                "result": clean_report,
                "document_metadata": _metadata(),
                "model_id": "MODEL-ALPHA",
            }
        )

        assert _status_ok(
            result
        ), f"Expected SUCCESS for clean result, got {result}. error_log: {result.get('error_log')}"
        assert result.get("result") == clean_report

    def test_empty_result_is_reported_as_an_error_not_a_silent_success(self):
        """An empty result is an ERROR carrying a notice, never an empty success.

        A success envelope whose output field is empty gives the caller no
        signal that anything went wrong; a refusal it can act on is strictly
        better.
        """
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        result = node.execute(
            {
                "result": "",
                "document_metadata": _metadata(),
                "model_id": "MODEL-ALPHA",
            }
        )

        _assert_withheld(result, "empty result")

    def test_missing_document_metadata_is_blocked(self):
        """Non-empty result with absent document_metadata must withhold output."""
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        result = node.execute(
            {
                "result": "# FSA Validation Document\n\nclean body",
                "document_metadata": {},
                "model_id": "MODEL-ALPHA",
            }
        )

        _assert_withheld(result, "missing document_metadata")


class TestModelIdResolution:
    """F-4: model_id is resolved from document_metadata / model_metadata, not a
    bare top-level state["model_id"] (which upstream nodes never set)."""

    def test_model_id_from_document_metadata(self, monkeypatch):
        """model_id surfaced in the audit trace must come from document_metadata."""
        import src.nodes.post_process_node as mod

        captured = {}
        monkeypatch.setattr(
            mod,
            "emit_trace_event",
            lambda event, payload, st: captured.update(payload),
        )

        node = mod.PostProcessNode()
        result = node.execute(
            {
                "result": "# FSA Validation Document\n\nclean body",
                "document_metadata": _metadata(),  # carries model_id MODEL-ALPHA
                # NOTE: no top-level state["model_id"] — proves it is read from metadata
            }
        )

        assert _status_ok(result)
        assert (
            captured.get("model_id") == "MODEL-ALPHA"
        ), f"model_id must be resolved from document_metadata, got {captured.get('model_id')!r}"

    def test_model_id_from_model_metadata_fallback(self, monkeypatch):
        """When document_metadata lacks model_id, fall back to model_metadata."""
        import src.nodes.post_process_node as mod

        captured = {}
        monkeypatch.setattr(
            mod,
            "emit_trace_event",
            lambda event, payload, st: captured.update(payload),
        )

        meta_no_model = {k: v for k, v in _metadata().items() if k != "model_id"}
        node = mod.PostProcessNode()
        result = node.execute(
            {
                "result": "# FSA Validation Document\n\nclean body",
                "document_metadata": meta_no_model,
                "model_metadata": {"model_id": "MODEL-GAMMA"},
            }
        )

        assert _status_ok(result)
        assert captured.get("model_id") == "MODEL-GAMMA"


class TestTC08PostProcessNodeTrustLevel:
    """TC-08: PostProcessNode.required_trust_level = TrustLevel.VERIFIED_EXTERNAL."""

    def test_post_process_node_trust_level(self):
        """PostProcessNode.required_trust_level must be VERIFIED_EXTERNAL."""
        from src.nodes.post_process_node import PostProcessNode
        from framework.schemas.trust_level import TrustLevel

        assert (
            PostProcessNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL
        ), f"Expected VERIFIED_EXTERNAL, got {PostProcessNode.required_trust_level}"


class TestTC10S3BlocksJwtAndBearer:
    """TC-10: the output gate blocks JWT and Bearer-token patterns."""

    def test_jwt_in_result_is_blocked(self):
        """result containing a JWT pattern must return ERROR."""
        from src.nodes.post_process_node import PostProcessNode

        jwt_pattern = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6Ikpva"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        node = PostProcessNode()
        result = node.execute(
            {
                "result": f"Validation document with leaked token: {jwt_pattern}",
                "document_metadata": _metadata(),
                "model_id": "MODEL-ALPHA",
            }
        )

        _assert_withheld(result, "jwt")

    def test_bearer_token_in_result_is_blocked(self):
        """result containing a Bearer token pattern must return ERROR."""
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        result = node.execute(
            {
                "result": "Auth header leaked: Bearer abcdefghijklmnopqrstuvwxyz0123456789",
                "document_metadata": _metadata(),
                "model_id": "MODEL-ALPHA",
            }
        )

        _assert_withheld(result, "bearer token")

    def test_clean_document_passes_gate(self):
        """A clean validation document is not withheld."""
        from src.nodes.post_process_node import PostProcessNode

        node = PostProcessNode()
        result = node.execute(
            {
                "result": "# FSA Scorecard\n\nMODEL-ALPHA passed all required checks.",
                "document_metadata": _metadata(),
                "model_id": "MODEL-ALPHA",
            }
        )

        assert _status_ok(result)
