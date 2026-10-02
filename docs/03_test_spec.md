# Test Specification — FIN-C2-051

**Template:** AlgorithmicTradingBacktestingAgent
**Category:** Cat 2 (domain-specific document-generation pipeline)

The suite is organised in three layers, and each answers a question the others cannot:

- **Unit** — what a single node returns, called directly. A direct call is the only way to
  observe *which* layer refused something, because the framework's own gates sit in front of
  every node result and refuse the same shapes.
- **Integration** — what the deployed HTTP entry point answers. The trust gate runs inside the
  framework's node wrapper, and the validated payload only crosses into the inner graph during
  a real invocation, so neither is exercised by a direct `execute()` call at all.
- **Proof of boundary** — the properties that must hold at a boundary regardless of the
  domain: the audit trail, checkpoint safety, import isolation, invocation order, the trust
  lattice, and output containment.

---

## Framework compliance

| TC-ID | Name | Target | Test input | Expected outcome | File |
|-------|------|--------|-----------|------------------|------|
| TC-01 | State is a flat TypedDict | `src/schemas/state.py` | Instantiate `State` with JSON-serializable primitives | `State` subclasses `AgentState`; no Pydantic | `tests/unit/test_state.py` |
| TC-02 | An empty request is refused | `PreProcessNode` | `user_input = ""` | `status == ERROR`; the message names the field | `tests/unit/test_graph.py`, `tests/integration/test_invoke_contract.py` |
| TC-03 | No credential fields in state | `src/schemas/state.py` | Inspect the field names | No field named password / token / api_key / secret | `tests/unit/test_state.py` |
| TC-05 | One audit event per node | every `FunctionNode` subclass | Execute each node once with `emit_trace_event` patched | Exactly one distinct event name per `execute()` | `tests/proof_of_boundary/test_pb_fin_c2_051.py` |
| TC-06 | The default input gate cannot be overridden | `FunctionNode` subclass | Define a subclass overriding `_security_gate_input` | `TypeError` at class definition | `tests/unit/test_framework_compliance_tc06_tc07.py` |
| TC-07 | The output gate blocks an API key | `PostProcessNode` | `result` carries a synthetic provider key (`sk-` followed by a run of the alphabet) | `status == ERROR`; every output-bearing field cleared; a truthy notice set | `tests/unit/test_post_process_node.py` |
| TC-08 | The declared trust level is the one every node requires | every node | Inspect the class attributes against the manifest | No node exceeds `required_trust_level` from `config/agent.yaml` | `tests/proof_of_boundary/test_pb_trust_lattice.py` |
| TC-09 | The document is non-empty on valid metrics | `DocumentGenerationNode` | Valid performance / risk metrics plus check results | `validation_document` is a non-empty markdown document with all five sections | `tests/unit/test_document_generation_node.py` |
| TC-10 | The output gate blocks JWT and Bearer shapes | `PostProcessNode` | `result` carries a JWT-shaped and a Bearer-shaped string | `status == ERROR`; output withheld | `tests/unit/test_post_process_node.py` |
| TC-11 | Every domain node emits its audit event | all five `FunctionNode` subclasses | Execute each node with the emitter patched | A distinct event name, with a dict payload, per node | `tests/unit/test_backtest_ingestion_node.py`, `tests/proof_of_boundary/test_pb_fin_c2_051.py` |

---

## Proof-of-boundary tests

| PB-ID | Boundary | What it proves | File |
|-------|----------|------|------|
| PB-1 | BaseNode → event emitter | Every domain node's audit event fires exactly once on the success path | `tests/proof_of_boundary/test_pb_fin_c2_051.py` |
| PB-2 | State serialization | Post-invoke state fields are primitives only — msgpack-safe | `tests/proof_of_boundary/test_pb_fin_c2_051.py` |
| PB-3 | External service | The ingestion node's injectable data-source interface is exercised (reference series and injected adapter) | `tests/proof_of_boundary/test_pb_fin_c2_051.py` |
| PB-4 | Import isolation | No platform-internal SDK imports under `src/` — AST scan of every `.py` file | `tests/proof_of_boundary/test_import_isolation.py` |
| PB-5 | Checkpoint safety | Serialized state carries no JWT-shaped values and no non-primitive objects | `tests/proof_of_boundary/test_pb_fin_c2_051.py`, `tests/proof_of_boundary/test_state_safety.py` |
| PB-6 | Invocation order | `__call__()` runs the trust gate → `node_start` → input gate → `execute()` → output gate → `node_complete`, and an under-privileged caller is denied before `execute()` | `tests/proof_of_boundary/test_pb_invoke_order.py` |
| PB-7 | Human-review interrupt propagation | Conditional: applies only when `config/config.yaml` enables human review. This template does not, so the module reports a skip with that reason | `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | Trust lattice | No node demands a level above the manifest's declared entry level, and a caller holding exactly that level receives a document | `tests/proof_of_boundary/test_pb_trust_lattice.py` |
| PB-9 | Output containment | The error envelope carries neither the tainted value, nor the document, nor a traceback, nor a source path — with a clean-path control proving a refuse-everything gate cannot pass | `tests/proof_of_boundary/test_pb_output_containment.py` |

> **The audit API is the free function** `from shared.utils.audit_logger import
> emit_trace_event`. `self.emit_trace_event(...)` does not exist on `FunctionNode` and raises
> `AttributeError`. The import resolves once the framework is installed; a bare checkout
> skips those cases rather than failing them.

---

## Caller-contract tests (integration)

| Case | What it pins | File |
|---|---|---|
| Missing / wrong Bearer token → 401 | The deployed entry point is reachable by the caller the manifest declares, and by no one else. Without this boundary every request arrives anonymous, the trust gate denies it, and the agent answers an error to every call while the unit suite stays green | `tests/integration/test_invoke_contract.py` |
| Authenticated request → a document | The public path does real work end to end | `tests/integration/test_invoke_contract.py` |
| The requested window renders in the header | The output depends on what the caller asked for | `tests/integration/test_invoke_contract.py` |
| Two different record series → two different reports | The report is computed from the caller's data, not fixed | `tests/integration/test_invoke_contract.py` |
| No performance row renders N/A or None | The renderer reads the keys the metrics node writes | `tests/integration/test_invoke_contract.py` |
| Absent records → the reference series, named in the header | The fallback is documented in the output, so a report cannot be mistaken for one computed from real data | `tests/integration/test_invoke_contract.py` |
| Non-finite matrix per field (`NaN`, `Infinity`, `-Infinity`, numeric string, bool, over-magnitude) | Every caller-controlled number fails closed. `NaN` is the dangerous case: it survives raw JSON and compares false against everything, so an unvalidated one turns a regulatory check into a silent pass | `tests/integration/test_invoke_contract.py` |
| Malformed structures (bad dates, wrong container types, missing required field) | The structured channel is validated, not trusted | `tests/integration/test_invoke_contract.py` |
| The record cap at its exact boundary (at cap accepted, one over refused) | The documented cap is the enforced cap | `tests/integration/test_invoke_contract.py` |
| A rejection never echoes the rejected value | Caller text does not travel into the error path | `tests/integration/test_invoke_contract.py` |
| Credential-shaped structured parameters → 400 naming the field | A credential-shaped value otherwise kills the run at the first node with nothing naming the cause. Declared-key contracts are not immunity: validators ignore undeclared keys, and ignoring is not stripping | `tests/integration/test_invoke_contract.py` |
| `refused == bool(detect_credentials_in_value(context))` | Per-field screening composes exactly to scanning the mapping, so the screen cannot drift away from the gate | `tests/integration/test_invoke_contract.py` |
| Oversized structured parameters → 413 | The coarse adapter guard | `tests/integration/test_invoke_contract.py` |
| A declared threshold changes the verdict | A value written in `config/config.yaml` reaches the node that reads it. Building the graph without its configuration reads as harmless — nothing fails until a declared value is expected to matter | `tests/integration/test_invoke_contract.py` |
| A malformed declared threshold falls back to the default | A bad declaration must not disable the check it feeds | `tests/integration/test_invoke_contract.py` |
| The deployed agent holds the declared configuration, and the threshold reaches the node instance | Loading, forwarding and handing over are three separate steps; a break in any one leaves the declaration inert | `tests/integration/test_invoke_contract.py` |

---

## Containment tests

| Case | What it pins | File |
|---|---|---|
| Each credential shape the framework detector knows, injected on the DATA path through the ingestion node's documented adapter | Neither the tainted value nor any part of the document reaches the caller. The fault is injected on the data path, never by patching a gate — patching the gate tests the patch | `tests/proof_of_boundary/test_pb_output_containment.py` |
| No traceback and no source path in the envelope | An error surface is not a debugging surface | `tests/proof_of_boundary/test_pb_output_containment.py` |
| The withheld notice is truthy | The framework projects `formatted_output or result`; a falsy notice re-activates the fallback onto `result` | `tests/proof_of_boundary/test_pb_output_containment.py` |
| Clean-path control: the same request still returns its real answer, and the gate node appears in the trace | A refuse-everything gate cannot pass the containment tests, and the answer is released *by* the gate rather than around it | `tests/proof_of_boundary/test_pb_output_containment.py` |
| Both gates clear rather than omit — presence AND emptiness asserted | LangGraph merges partial deltas, so an omitted key keeps its old value. `assert not delta.get(field)` would pass on a gate that clears nothing | `tests/proof_of_boundary/test_pb_output_containment.py`, `tests/unit/test_post_process_node.py` |
| Every framework-known shape is caught locally too | A local pattern set narrower than the framework's is a bypass: the framework raises and the wrapper discards the node's delta, clearing included | `tests/proof_of_boundary/test_pb_output_containment.py`, `tests/unit/test_document_generation_node.py` |
| Ordinary prose using credential vocabulary is still released | The gate must not fire on legitimate domain wording | `tests/proof_of_boundary/test_pb_output_containment.py`, `tests/unit/test_document_generation_node.py` |
| No violation message carries the matched value | The framework scans every returned value; an echoed credential would make it raise and discard the clearing | `tests/proof_of_boundary/test_pb_output_containment.py` |

---

## Business-logic tests

| BL-ID | Name | Scenario | Expected result | File |
|-------|------|--------------|-----------------|------|
| BL-01 | Sharpe ratio from a known series | A return series with a known mean and standard deviation | `performance_metrics["sharpe_ratio"]` matches the annualised value | `tests/unit/test_validation_metrics_node.py` |
| BL-02 | Drawdown depth and duration | A series with an embedded peak-to-trough drawdown | `max_drawdown_pct` and `max_drawdown_duration_bars` match the expected depth and length | `tests/unit/test_validation_metrics_node.py` |
| BL-03 | FSA-CHK-01 fails below the threshold | Sharpe below the effective minimum | The check entry reports `FAIL` with the threshold and the actual value | `tests/unit/test_validation_metrics_node.py` |
| BL-04 | FSA-CHK-03 fails on a short window | A window shorter than the effective minimum | The check entry reports `FAIL` | `tests/unit/test_validation_metrics_node.py` |
| BL-05 | The document carries every section | A full inner-pipeline run | The document contains all five section headers, the check-results table and the metrics tables | `tests/unit/test_document_generation_node.py` |

---

## Test file map

```
tests/
  unit/
    test_state.py                          — TC-01, TC-03
    test_graph.py                          — graph composition, extract_input/merge_output, the bridge
    test_backtest_ingestion_node.py        — record-source precedence, TC-11
    test_validation_metrics_node.py        — BL-01..BL-04, TC-05
    test_document_generation_node.py       — BL-05, TC-09, the document gate
    test_post_process_node.py              — TC-07, TC-10, the response gate
    test_framework_compliance_tc06_tc07.py — TC-06, TC-07 (gate override refusal)
  integration/
    test_end_to_end.py                     — the four nodes threaded in sequence
    test_invoke_contract.py                — the deployed /invoke contract
  proof_of_boundary/
    test_import_isolation.py               — PB-4
    test_state_safety.py                   — PB-5
    test_pb_fin_c2_051.py                  — PB-1, PB-2, PB-3, PB-5
    test_pb_invoke_order.py                — PB-6
    test_pb7_hitl_interrupt_propagation.py — PB-7 (skips: human review is not enabled)
    test_pb_trust_lattice.py               — PB-8
    test_pb_output_containment.py          — PB-9
```

---

## Verifying the suite is load-bearing

A passing suite says what the tests cover, not that the code works. Each guard was therefore
reverted individually and the suite re-run; every revert must produce failures, and the
number of failures is what shows which layer the guard holds up. The mutant that matters most
is the last one — restoring the original shipped `src/` — because layered guards mask each
other, and removing them one at a time can leave a full-invocation test green while the test
itself is decorative.

When the original source is restored, the new test modules may reference symbols that source
does not define. pytest then reports a collection **error**, not a failure, and a naive
"did anything fail?" check reads that as a pass. Re-run a suite that imports cleanly against
the original and read the actual pytest summary line.

---

## Coverage targets

- `src/nodes/`: ≥ 80% line coverage.
- `src/graph/`: `register_nodes()`, `extract_input()`, `merge_output()`,
  `_extra_initial_state()` and the configuration validation fully covered.
- `src/api/server.py`: the authentication boundary, the size cap and the credential screen
  each covered in both directions.
