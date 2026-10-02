# FIN-C2-051 — AlgorithmicTradingBacktestingAgent

> **Category**: Cat 2 (domain-specific document-generation pipeline)
> **Industry**: FIN (financial services)

## Overview

Turns the record series from an algorithmic trading backtest into a formal model validation document in the shape a financial regulator expects. The caller submits the model identifier, the asset class, the validation window and the per-period return series; the pipeline computes the standard performance and risk statistics (Sharpe, Sortino, Calmar, annualised return, win rate, maximum drawdown and its duration, 95% VaR and CVaR, annualised volatility, beta), runs five regulatory model-risk checks against configurable thresholds, and renders a five-section document: executive summary, performance metrics, risk metrics, regulatory check results, and conclusions with remediation guidance.

Generation is deterministic. Every sentence is rendered from the computed numbers — no language model is called anywhere in the path — so the same records always produce the same document and each figure in the report is traceable to the series it came from. When no record series is supplied the pipeline runs on a small built-in reference series and says so in the document header, so a report can never be mistaken for one computed from real data.

Intended users are quantitative analysts, model risk officers and algorithmic trading desks who have to produce a validation artefact for each model revision.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/graph/    outer graph, inner domain workflow graph, and the payload bridge between them
src/nodes/    input validation, record ingestion, metric computation, document assembly, output gate
src/schemas/  the shared state definition
src/api/      the HTTP entry point
tests/        unit, integration and boundary tests
config/       agent.yaml (the manifest) and config.yaml (runtime parameters)
docs/         design specification and test specification
```

See `docs/02_design.md` for the node-by-node design and `docs/03_test_spec.md` for the test
specification.

## Invoking the agent

`POST /invoke` takes the request as a JSON string in `input`, and the record series as
structured parameters in `input_context`:

```jsonc
{
  "input": "{\"model_id\": \"MODEL-ALPHA\", \"asset_class\": \"equity\", \"backtest_data\": \"inline:reference\", \"validation_date_start\": \"2022-01-01\", \"validation_date_end\": \"2024-12-01\"}",
  "input_context": {
    "backtest_records": [
      {"date": "2022-01-01", "portfolio_return": 0.01, "benchmark_return": 0.008}
    ],
    "oos_months": 8,
    "registration_date": "2025-01-01"
  }
}
```

Every caller-supplied number goes through a finite and bounded parser, so `NaN`, `Infinity`
and out-of-range values are refused rather than silently compared (a comparison against `NaN`
is always false, which would turn a regulatory check into a silent pass). Rejections name the
offending field and never repeat the rejected value.

In a standalone deployment, set `INVOKE_AUTH_TOKEN` in the server environment and present it
as `Authorization: Bearer <token>`; requests that no upstream middleware has vouched for are
otherwise anonymous and the agent's entry node refuses them.

## Customising

1. Adjust `config/config.yaml` — the regulatory thresholds are declared there and are validated
   for type, finiteness and range before they reach the code that reads them.
2. Point `BacktestIngestionNode` at your own data source by passing an adapter that exposes
   `load(model_id, start_date, end_date, asset_class)`.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
