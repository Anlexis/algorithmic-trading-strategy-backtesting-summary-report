# FIN-C2-051 — Unit Tests: State schema (TC-01, TC-03)
#
# TC-01: State is a flat TypedDict subclass of AgentState (no Pydantic)
# TC-03: No credential-bearing field names in State
#
# Mirrors the peer template #11 idiom: TypedDicts forbid issubclass(), so the
# TypedDict-safe equivalent is to assert every AgentState field is present in
# State's annotations and that no field uses a Pydantic/dataclass type.

import typing


class TestTC01StateFlatTypedDict:
    """TC-01: State(AgentState) is a flat TypedDict; no Pydantic/dataclass."""

    def test_state_carries_all_agentstate_fields(self):
        """State must extend AgentState — i.e. carry every AgentState field.

        State is a TypedDict, and TypedDicts forbid issubclass() (TypeError).
        The TypedDict-safe equivalent is to assert that every field declared on
        AgentState is also present in State's annotations (State = base fields +
        its own FIN-C2-051 fields).
        """
        from src.schemas.state import State
        from framework.schemas.agent_state import AgentState

        base_fields = set(AgentState.__annotations__)
        state_fields = set(State.__annotations__)
        missing = base_fields - state_fields
        assert not missing, "State must carry all AgentState fields; missing: " + ", ".join(sorted(missing))

    def test_state_declares_fin_c2_051_fields(self):
        """State must declare its own FIN-C2-051 outer + inner fields."""
        from src.schemas.state import State

        state_fields = set(State.__annotations__)
        expected = {
            # outer layer
            "validated_input",
            "result",
            # BacktestIngestionNode output
            "backtest_results",
            "model_metadata",
            "benchmark_data",
            # ValidationMetricsNode output
            "performance_metrics",
            "risk_metrics",
            "fsa_checks",
            # DocumentGenerationNode output
            "validation_document",
            "llm_response",
            "document_metadata",
        }
        missing = expected - state_fields
        assert not missing, "State must declare FIN-C2-051 fields; missing: " + ", ".join(sorted(missing))

    def test_state_has_no_pydantic_base(self):
        """State must NOT be a Pydantic BaseModel subclass."""
        from src.schemas.state import State

        try:
            from pydantic import BaseModel

            assert not issubclass(State, BaseModel), "State must NOT inherit from Pydantic BaseModel"
        except ImportError:
            pass  # Pydantic not installed — isolation confirmed
        except TypeError:
            # TypedDicts forbid issubclass(); the type error itself proves State
            # is a TypedDict, not a Pydantic BaseModel.
            pass

    def test_state_fields_not_pydantic_or_dataclass(self):
        """All State field annotations must avoid Pydantic BaseModel / dataclass types."""
        from src.schemas.state import State

        hints = typing.get_type_hints(State)
        for field_name, field_type in hints.items():
            type_str = str(field_type)
            assert "BaseModel" not in type_str, f"Field {field_name}: must not use Pydantic BaseModel, got {field_type}"
            assert "dataclass" not in type_str.lower(), f"Field {field_name}: must not use dataclass, got {field_type}"

    def test_state_has_required_outer_fields(self):
        """State must define validated_input and result (outer layer fields)."""
        from src.schemas.state import State

        hints = typing.get_type_hints(State)
        assert "validated_input" in hints, "State must have validated_input field"
        assert "result" in hints, "State must have result field"

    def test_state_has_required_inner_fields(self):
        """State must define the BacktestIngestionNode output fields."""
        from src.schemas.state import State

        hints = typing.get_type_hints(State)
        for field in ("backtest_results", "model_metadata", "benchmark_data"):
            assert field in hints, f"State must have {field} field"

    def test_state_has_metrics_and_document_fields(self):
        """State must define metrics + FSA + document fields."""
        from src.schemas.state import State

        hints = typing.get_type_hints(State)
        for field in (
            "performance_metrics",
            "risk_metrics",
            "fsa_checks",
            "validation_document",
            "llm_response",
            "document_metadata",
        ):
            assert field in hints, f"State must have {field} field"


class TestTC03NoCredentialsInState:
    """TC-03: State must not have fields with credential-bearing names."""

    def test_no_credential_field_names(self):
        """State fields must not include password, token, api_key, secret, etc."""
        from src.schemas.state import State

        hints = typing.get_type_hints(State)
        forbidden_substrings = (
            "password",
            "passwd",
            "token",
            "api_key",
            "secret",
            "private_key",
            "access_key",
            "credentials",
        )
        violations = []
        for field_name in hints:
            for bad in forbidden_substrings:
                if bad in field_name.lower():
                    violations.append(f"Field '{field_name}' matches forbidden pattern '{bad}'")

        assert not violations, "State has credential-bearing field names:\n" + "\n".join(violations)
