# ToolCallingAgent vs CodeAgent — Parameter Sweep

A local experiment harness comparing two smolagents agent architectures on a chained multi-step task.

## What it does

Runs a temperature sweep across `ToolCallingAgent` and `CodeAgent` on the same task:
1. Web search for top Seattle pizza restaurants
2. Call `get_price_tier(restaurant)` for each result
3. Call `pizzas_in_budget(tier, budget)` to chain the output
4. Produce a structured final answer

Measures latency, success rate, and tool-routing accuracy per agent type across temperatures.

## Key finding

`ToolCallingAgent` makes one JSON tool call per step — needs `max_steps ≥ 12` for this task.  
`CodeAgent` generates Python loops — batches all calls in one step, efficient but brittle at extreme temperatures.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
HF_TOKEN=your_token_here
OTEL_ENABLED=1
```

Requires [Ollama](https://ollama.com) running locally with `qwen2.5-coder:14b` pulled.

## Run

```bash
SWEEP_PARAMS=temperature TEMP_VALUES=0.0,0.3,0.7 SWEEP_RUNS=5 \
MODEL_TIMEOUT_SECONDS=60 python login.py
```

Results write to `metrics_output/` after every completed run pair — safe to Ctrl+C at any time.

## Env vars

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `ollama/qwen2.5-coder:14b` | Model to use |
| `SWEEP_PARAMS` | `temperature` | Comma-separated params to sweep |
| `TEMP_VALUES` | `0.0,0.1,0.3,0.6` | Temperature values |
| `SWEEP_RUNS` | `2` | Runs per config |
| `MODEL_TIMEOUT_SECONDS` | `90` | Per-request timeout |
| `OTEL_ENABLED` | `0` | Enable OpenTelemetry tracing |
