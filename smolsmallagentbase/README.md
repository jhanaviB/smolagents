# ToolCallingAgent vs CodeAgent — Compare both with a local LLM

This code contains a quick experiment (n = 10) to compare a ToolCallingAgent with a CodeAgent. Both the agents are run on a 16GB base M4 mac model.
Given how constrained the hardware is, I did not expect either of the agents to perform too well. I was more interested in seeing _how_ things failed instead of ascertain when.

The agents were given a web search tool and deterministic functions. I wanted to see if the agent was smart enough to understand when to use the deterministic functions and how varying temp or the agent type changed that.

## What it does

Runs a temperature sweep across `ToolCallingAgent` and `CodeAgent` on the same task:
1. Web search to find the best pizza places in Seattle!
2. Call `get_price_tier(restaurant)` for each result. (This function is a deterministic function to find the price tier of restaurants. It computes this via hashing the restaurant's name).
3. Call `pizzas_in_budget(tier, budget)` thereafter to chain the output
4. Produce a structured final answer

It measures latency, success rate, and tool-routing accuracy per agent type across temperatures.

## Key finding

`ToolCallingAgent` makes one JSON tool call per step. This performed _ok_ in all cases. Around ~40% accuracy, i.e it only routed correctly for 2 restaurants out of 5 Never a stellar performance though!

`CodeAgent` generates Python loops and batches all calls in one step. This is efficient but brittle since it has the chance of the max blast radius. With slight varance in temperature, the probability of python syntax getting jumbled up or logic being wrong causes the agent to get into infinite loops that my LLM model couldn't handle. In the experiements I ran, it was able to route with 90% accuracy. 

`ToolCallingAgent` fabricated prices for 2/5 restaurants. 

How your local setup performs depends heavily depends on the task. A well-defined system prompt goes a long way. While I had the best results with CodeAgents, their high blast radius can make them unreliable in some cases. Next, I want to see the results of combining deterministic validation steps with a coding agent.
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

## Env vars

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `ollama/qwen2.5-coder:14b` | Model to use |
| `SWEEP_PARAMS` | `temperature` | Comma-separated params to sweep |
| `TEMP_VALUES` | `0.0,0.1,0.3,0.6` | Temperature values |
| `SWEEP_RUNS` | `2` | Runs per config |
| `MODEL_TIMEOUT_SECONDS` | `90` | Per-request timeout |
| `OTEL_ENABLED` | `0` | Enable OpenTelemetry tracing |

