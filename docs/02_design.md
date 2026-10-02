# Template Design Specification — FIN-C2-051

**Template:** AlgorithmicTradingBacktestingAgent
**Category:** Cat 2 (domain-specific document-generation pipeline)
**Industry:** FIN (finance / securities)

## Position in the AgentCore architecture

- **Agent class:** `FinC2051Agent(AgentBaseGraph)` — the outer graph (`src/graph/graph.py`),
  also exported as `Graph`.

| Layer | Class |
|---|---|
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |

- **Pattern:** nested Cat 2 — the fixed outer backbone with an inner `DomainWorkflowGraph`
  occupying the `main` slot.
- **Three-layer separation:**
  - State: `State(AgentState)` — a flat TypedDict; no Pydantic (msgpack-safe).
  - Nodes: `FunctionNode` subclasses; `execute(self, state) -> dict`, returning a partial dict.
  - Graph: composition in `register_nodes()`; `add_edges()` is not overridden on the outer graph.

> **`execute()` takes one argument.** The framework's node wrapper calls
> `self.execute(state)`. A node written as `execute(self, state, config=None)` never receives
> a second argument, so every value read from it is silently the default — which is how a
> declared runtime setting becomes dead without any test noticing. Configuration reaches a
> node through its constructor instead (see *Runtime configuration* below).

---

## Architecture overview

### Outer backbone

| Node | Class | Responsibility | Input state fields | Output state fields | Parent class |
|------|-------|---------------|-------------------|---------------------|-------------|
| initialize | `InitializeNode` (framework default) | Sets schema_version, session_id, caller trust level | user_input | session_id, caller_trust_level | InitializeNode |
| pre_process | `PreProcessNode` | Validates the request and the structured caller parameters; builds the validated payload | user_input, input_context | validated_input | FunctionNode |
| main | `FinC2051GraphNode` | Delegates the domain workflow to `DomainWorkflowGraph` and bridges the validated payload across the boundary | validated_input | result, validation_document, document_metadata, llm_response | GraphNode |
| post_process | `PostProcessNode` | The response output gate: scans the released report and withholds it on a violation | result, document_metadata | result, formatted_output (when withheld) | FunctionNode |
| finalize | `FinalizeNode` (framework default) | Builds response_metadata, total_time_ms | status | response_metadata | FinalizeNode |

### Inner `DomainWorkflowGraph`

| Node | Class | Responsibility | Input state fields | Output state fields | Parent class |
|------|-------|---------------|-------------------|---------------------|-------------|
| backtest_ingestion | `BacktestIngestionNode` | Assembles the record series: the caller's validated series, else an injected data-source adapter, else the built-in reference series | validated_input | backtest_results, model_metadata, benchmark_data, result | FunctionNode |
| validation_metrics | `ValidationMetricsNode` | Computes the performance and risk statistics and runs the five regulatory checks against the effective thresholds | backtest_results, benchmark_data, model_metadata, validated_input | performance_metrics, risk_metrics, fsa_checks | FunctionNode |
| document_generation | `DocumentGenerationNode` | Renders the five-section validation document from the computed numbers, then applies the document output gate | performance_metrics, risk_metrics, fsa_checks, model_metadata, validated_input | validation_document, llm_response, result, document_metadata | FunctionNode |

### Data flow

```
START
  │
  ▼
InitializeNode        — session_id, caller trust level
  │
  ▼
PreProcessNode        — validate model_id / asset_class / window / record series
  │                     writes validated_input
  ▼
FinC2051GraphNode     — extract_input() stashes the payload on the bridge,
  │                     then invokes DomainWorkflowGraph
  │
  │   ┌──────────────────────────────────────────────────────────────┐
  │   │ DomainWorkflowGraph (inner)                                  │
  │   │   _extra_initial_state() seeds validated_input from the      │
  │   │   bridge — the framework forwards nothing else               │
  │   │                                                              │
  │   │   BacktestIngestionNode                                      │
  │   │     emit_trace_event('backtest_ingestion_complete')          │
  │   │        ↓                                                     │
  │   │   ValidationMetricsNode                                      │
  │   │     emit_trace_event('validation_metrics_calculated')        │
  │   │        ↓                                                     │
  │   │   DocumentGenerationNode                                     │
  │   │     emit_trace_event('document_generation_complete')         │
  │   └──────────────────────────────────────────────────────────────┘
  │
  │                     merge_output() maps the inner result back
  ▼
PostProcessNode       — output gate on `result`; withholds on a violation
  │                     emit_trace_event('post_process_complete')
  ▼
FinalizeNode          — response_metadata
  │
  ▼
END
```

### The outer→inner payload bridge

The framework invokes an inner graph as `subgraph.invoke(user_input, session_id=..., ctx=...)`.
Whatever `extract_input()` returns therefore lands on the inner graph's `user_input` key and
**nowhere else** — the outer state's `validated_input` and `input_context` are not forwarded.
The inner domain nodes read `validated_input`.

`src/graph/context_bridge.py` carries the payload across, through the two sanctioned hooks:

| Step | Hook | What it does |
|---|---|---|
| Before the inner invoke | `FinC2051GraphNode.extract_input(state)` | `set_caller_payload(<validated payload>)` |
| Inside the inner invoke | `DomainWorkflowGraph._extra_initial_state()` | returns `{"validated_input": get_caller_payload()}` |

A `ContextVar` holds the hand-off, so concurrent invocations in one process cannot observe
each other's payload. Without the bridge every real invocation reaches
`BacktestIngestionNode` with no payload and the request is refused — while a unit suite that
calls `execute()` directly stays entirely green.

---

## The caller contract

`POST /invoke` (`src/api/server.py`) takes:

| Field | Channel | Contract |
|---|---|---|
| `input` | JSON string | `model_id` (inert identifier, `[A-Za-z0-9_-]{1,64}`), `asset_class` (closed allowlist), `backtest_data` (reference or inline data, capped at 10 MB), `validation_date_start` / `validation_date_end` (ISO dates; the end date must not be in the future) |
| `input_context.backtest_records` | structured | Up to 5,000 period records, each `{date, portfolio_return, benchmark_return, cumulative_return}`; `portfolio_return` is required per record |
| `input_context.oos_months` | structured | Out-of-sample window, 0–600 whole months |
| `input_context.registration_date` | structured | ISO date the model was registered |

**Every caller-controlled number goes through a finite and bounded parser.** Booleans,
strings, non-numerics, `NaN`, `±Infinity` and out-of-range magnitudes are all refused.
The non-finite case is the one that matters: `NaN` survives raw JSON intact and every
comparison against it is false, so an unvalidated `NaN` return would make each regulatory
threshold read as "not violated" — a silent pass on the exact decision this agent exists to
make.

**Rejections name the field, never the value.** Caller text is not echoed into `error_log`.

**The only caller string that renders into the document is `model_id`**, and it is locked to
an inert identifier alphabet. The window dates render too, and they are normalised through a
strict ISO parse first. There is no free-text channel into the output, so the report cannot
be used as an output-injection surface.

### Structured-parameter credential screen

The framework's mandatory output gate scans every value of every node result for credential
patterns, and the backbone's first node copies the structured parameters verbatim into its
own result. A credential-shaped string anywhere in them therefore makes the **first** node of
the graph fail before any template code runs, and the caller receives an error naming
nothing.

Declaring an inert contract is not immunity: validators *ignore* undeclared keys, and
ignoring is not stripping. The adapter therefore screens the whole mapping as submitted,
using the framework's own `detect_credentials_in_value`, and refuses with `400` naming the
offending field. Scanning field by field composes exactly to scanning the mapping (the
detector on a mapping is the union over its values), which is what lets the refusal name a
field without widening or narrowing the block set.

### Caller authentication

The entry node declares `VERIFIED_EXTERNAL`, and nothing establishes a trust level in a
standalone deployment. When `INVOKE_AUTH_TOKEN` is set on the server environment, a caller
that no upstream middleware vouched for must present it as a Bearer token and then runs at
`VERIFIED_EXTERNAL`; middleware-established trust is never demoted. Without that boundary
every deployed request arrives anonymous, the trust gate denies it, and the agent answers an
error to every single call.

### Trust levels

| Node | Required trust level |
|---|---|
| `PreProcessNode` | VERIFIED_EXTERNAL |
| `BacktestIngestionNode` | VERIFIED_EXTERNAL |
| `ValidationMetricsNode` | VERIFIED_EXTERNAL |
| `DocumentGenerationNode` | VERIFIED_EXTERNAL |
| `PostProcessNode` | VERIFIED_EXTERNAL |

The manifest declares `required_trust_level: VERIFIED_EXTERNAL`, and the trust gate is
ordered ANONYMOUS < VERIFIED_EXTERNAL < INTERNAL. A node demanding a level above the
manifest's entry level can never be reached by the caller the manifest describes, so no node
here exceeds it. None of these nodes holds a privileged capability: they read state, compute
and render.

---

## Runtime configuration

`config/agent.yaml` is the static manifest. `config/config.yaml` holds the runtime
parameters; the platform registry loads it and passes it as `Graph(config=...)`, and
`src/api/server.py` reads the same file through `src.graph.graph.runtime_config()` so both
deployments see identical configuration.

| Key | Read by | Bounds |
|---|---|---|
| `max_retry` | the framework backbone's routing | non-negative integer below the framework ceiling |
| `timeout_s` | the platform runtime | — |
| `fsa_thresholds.fsa_chk01_min_sharpe` | `ValidationMetricsNode` | −100 … 100 |
| `fsa_thresholds.fsa_chk02_max_drawdown_pct` | `ValidationMetricsNode` | 0 … 100 |
| `fsa_thresholds.fsa_chk03_min_backtest_months` | `ValidationMetricsNode` | 0 … 600 |
| `fsa_thresholds.fsa_chk04_min_oos_months` | `ValidationMetricsNode` | 0 … 600 |

Each threshold is validated for type, finiteness and range in `src/graph/graph.py` before it
is forwarded to the inner graph, which hands it to `ValidationMetricsNode`'s constructor. A
declaration that fails validation is dropped and the node keeps its documented built-in
default — a `NaN` threshold would otherwise compare false against everything and silently
disable the check it feeds.

---

## Regulatory checks

| Check | Rule | Threshold source |
|---|---|---|
| FSA-CHK-01 | Sharpe ratio ≥ minimum | `fsa_chk01_min_sharpe` (default 0.5) |
| FSA-CHK-02 | Maximum drawdown ≤ limit, per cent | `fsa_chk02_max_drawdown_pct` (default 30.0) |
| FSA-CHK-03 | Backtest window ≥ minimum, whole months | `fsa_chk03_min_backtest_months` (default 24) |
| FSA-CHK-04 | Out-of-sample window ≥ minimum, whole months | `fsa_chk04_min_oos_months` (default 6) |
| FSA-CHK-05 | The backtest window predates model registration (no data snooping) | — (fails closed when either date is absent) |

---

## Output invariant

The rendered document carries ratios, percentages and counts. **It renders no monetary
aggregates**, so no currency-precision grid applies and none is enforced; the invariant this
template does enforce at its output boundary is the credential one below.

### The output gate withholds by CLEARING

The framework projects an invocation's answer as `formatted_output or result`, with **no
status check**. Three consequences shape both gates:

1. A gate that returns an error without clearing `result` still ships the ungated answer
   inside the error envelope.
2. A falsy `formatted_output` *activates* that fallback, so `""` as a withheld-output marker
   produces the exact leak it was written to prevent. The notice is therefore truthy.
3. LangGraph merges partial deltas, so a key a node **omits** keeps whatever it already held.
   Clearing means returning the key with an empty value, not leaving it out.

On a violation each gate returns `AgentStatus.ERROR`, clears every output-bearing field, and
sets a truthy withheld-output notice. `DocumentGenerationNode` additionally asserts that
every output field it returns is one the gate knows how to clear, so a field added later
cannot quietly escape the clearing.

### Both gates use the framework's own detector

A local pattern set narrower than the framework's is not a second opinion, it is a bypass:
a value the framework catches and the template misses makes the framework raise inside its
own output gate, and the wrapper then returns a bare error partial that **discards the
node's whole delta — the clearing included**. Both gates therefore call
`detect_credentials_in_value`. `PostProcessNode` adds one form the framework does not carry,
the assignment shape (`password = …`, `token: …`), which is what a leaked configuration dump
looks like.

Violation messages name a field or a pattern class and never the matched text — the
framework scans every value a node returns, so an echoed credential would make it raise and
discard the clearing.

### An empty result is an error, not a quiet success

`PostProcessNode` reports an absent report as an error carrying a notice. A success envelope
whose output field is empty gives the caller no signal that anything was withheld or lost,
which is worse than a refusal it can act on.

---

## State definition

`src/schemas/state.py` — `State(AgentState)`, a flat TypedDict.

| Field | Type | Written by | Purpose |
|-------|------|----------|---------|
| validated_input | `Optional[Dict[str, Any]]` | PreProcessNode | The validated payload: model_id, asset_class, window, period, validation_date, backtest_records, oos_months, registration_date |
| backtest_results | `Optional[List[Dict[str, Any]]]` | BacktestIngestionNode | The period record series actually analysed |
| model_metadata | `Optional[Dict[str, Any]]` | BacktestIngestionNode | model_name, version, strategy_type, asset_class, window, data_source |
| benchmark_data | `Optional[List[Dict[str, Any]]]` | BacktestIngestionNode | Benchmark return series for the beta computation |
| performance_metrics | `Optional[Dict[str, Any]]` | ValidationMetricsNode | sharpe_ratio, sortino_ratio, calmar_ratio, annualized_return, win_rate_pct, max_drawdown_pct, max_drawdown_duration_bars |
| risk_metrics | `Optional[Dict[str, Any]]` | ValidationMetricsNode | var_95, cvar_95, annualized_volatility, beta, drawdown_periods |
| fsa_checks | `Optional[List[Dict[str, Any]]]` | ValidationMetricsNode | One entry per regulatory check: check_id, check_name, threshold, actual, result, detail |
| validation_document | `Optional[str]` | DocumentGenerationNode | The rendered document (markdown) |
| llm_response | `Optional[str]` | DocumentGenerationNode | The same rendered narrative, kept for the audit trail |
| document_metadata | `Optional[Dict[str, Any]]` | DocumentGenerationNode | model_id, generated_at, validation_date, pass/fail counts, record_count |
| result | `Optional[str]` | merge_output / PostProcessNode | The report the caller receives |
| trace_id, correlation_id | `Optional[str]` | framework | Correlation identifiers |

> **The renderer must read the keys the metrics node writes.** The performance table reads
> `max_drawdown_pct` and `win_rate_pct`; reading the shorter names renders those rows as N/A
> on every run whatever was computed, and nothing else fails. The same applies to the risk
> table's `var_95` / `cvar_95` / `annualized_volatility`. A metric that could not be computed
> is *present* with the value `None`, so `dict.get(key, "N/A")` never applies its default —
> the renderer maps `None` to `N/A` explicitly.

**State constraints:**
- Flat TypedDict only (primitives and JSON-serializable types).
- No credentials in state — checkpoints persist it.
- No Pydantic models, dataclasses or arbitrary Python objects (msgpack-incompatible).

---

## Framework utilisation

| Feature | Used | Notes |
|---------|------|-------|
| Domain input validation inside `PreProcessNode.execute()` | Yes | See *Security gate placement* below |
| Domain input gate inside `BacktestIngestionNode.execute()` | Yes | Model-identifier allowlist and maximum window |
| Document output gate inside `DocumentGenerationNode.execute()` | Yes | Empty-document check and credential scan over the rendered text and the metadata |
| Response output gate inside `PostProcessNode.execute()` | Yes | Credential scan over the released report |
| `_extra_security_gate_input` / `_extra_security_gate_output` hooks | **No (by design)** | See *Security gate placement* below |
| Audit logging (`emit_trace_event`, the free function) | Yes | Every domain node emits one event after its side-effect, via `from shared.utils.audit_logger import emit_trace_event`; the instance-method form does not exist |
| `error_strategy = "propagate"` | Yes | `FinC2051GraphNode` — an inner failure surfaces as `SubgraphError` |
| Inner `get_output()` surfaces `error_log` | Yes | The framework builds the `SubgraphError` from `sub_result["error_log"]`; omitting the key makes every inner failure surface as "failed: []" with no reason |

### Security gate placement — design decision

**Domain validation runs inside `execute()`; the `_extra_security_gate_input()` and
`_extra_security_gate_output()` hooks are deliberately not overridden.**

The framework auto-wraps those hooks into the graph chain. On the clean path a wrapped hook
returns `None`, the next node then receives `state=None`, and the invocation raises
`AttributeError`. Calling the same checks inline from `execute()` keeps every rule running
without that failure mode. Order is preserved either way: the framework's own gates run
around `execute()`, and the domain checks run at the top of it — before any side-effect and
before anything is returned.

---

## Design decision record

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base class | `AgentBaseGraph`, inherited directly | Framework policy for this template family |
| Category | Cat 2 (nested) | A multi-step domain workflow producing one deliverable |
| Inner graph composition | `DomainWorkflowGraph(BaseGraph)` behind a GraphNode | Domain complexity encapsulated; the backbone stays untouched |
| Payload hand-off | `ContextVar` bridge | The framework forwards only `user_input` into an inner graph |
| Metrics computation | Pure Python, no model call | Deterministic and auditable — a regulator has to be able to reproduce every figure |
| Document generation | Deterministic rendering from the computed numbers | `generation_mode: deterministic` in the manifest says so; the same records always produce the same document |
| Record source precedence | Caller series → injected adapter → built-in reference series | The public path computes from caller data; the reference series is a documented fallback, named in the document header so a report can never be mistaken for one computed from real data |
| Threshold configuration | `config/config.yaml`, validated before use | Regulators may require stricter thresholds per strategy type |
| Withheld output | Clear every output field, set a truthy notice | The framework projects `formatted_output or result` with no status check |
| Credential detection | The framework's own detector, plus the assignment form | A narrower local set is a containment bypass, not a second opinion |
