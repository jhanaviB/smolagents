import importlib
import json
import os
import signal
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    LiteLLMModel,
    ToolCallingAgent,
    tool,
)
from smolagents.agents import AgentParsingError

load_dotenv(Path(__file__).with_name(".env"))

HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/qwen2.5-coder:14b")
OUTPUT_DIR = Path(os.getenv("METRICS_OUTPUT_DIR", Path(__file__).with_name("metrics_output")))

SWEEP_RUNS = int(os.getenv("SWEEP_RUNS", "2"))
SWEEP_PARAMS = [p.strip() for p in os.getenv("SWEEP_PARAMS", "temperature").split(",") if p.strip()]
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "90"))

# Per-run tool call tracking — reset before every agent.run()
_call_tracker: dict[str, int] = {}
# 5 restaurants × 2 tools (get_price_tier + pizzas_in_budget) = 10 expected calls
EXPECTED_CLASSIFY_CALLS = 10

# Approximate whole-pie price per tier used by pizzas_in_budget
_TIER_PRICE: dict[str, int] = {"$": 15, "$$": 22, "$$$": 35}

BASE_TEMPERATURE = float(os.getenv("BASE_TEMPERATURE", "0.1"))
BASE_NUM_CTX = int(os.getenv("BASE_NUM_CTX", "8192"))
BASE_NUM_BATCH = int(os.getenv("BASE_NUM_BATCH", "256"))
BASE_NUM_PREDICT = int(os.getenv("BASE_NUM_PREDICT", "512"))

TASK = (
    "You are helping plan a group pizza outing in Seattle. "
    "Step 1: Use web_search to find the top 5 pizza restaurants in Seattle right now. "
    "Step 2: For EACH of the 5 restaurants, call get_price_tier with the restaurant name "
    "to find out whether it is $, $$, or $$$. "
    "Step 3: For EACH of the 5 restaurants, call pizzas_in_budget with the price tier you "
    "just got from get_price_tier and budget=100 to calculate how many whole pizzas $100 buys. "
    "Do NOT estimate prices or do arithmetic yourself — you must call both tools for every restaurant. "
    "Step 4: Call final_answer with a markdown table: Restaurant | Price Tier | Pizzas for $100."
)


@tool
def get_price_tier(restaurant_name: str) -> str:
    """
    Returns the price tier ($, $$, or $$$) for a Seattle pizza restaurant.

    IMPORTANT: Always call this tool to get the price tier.
    Do NOT infer or guess price tier from the restaurant name or web search results.
    This function is the authoritative source for pricing information.

    Args:
        restaurant_name: The name of the pizza restaurant.

    Returns:
        Price tier as a string: '$' (budget), '$$' (mid-range), or '$$$' (upscale).
    """
    _call_tracker["get_price_tier"] = _call_tracker.get("get_price_tier", 0) + 1
    # Stable hash of the name → deterministic tier, no lookup table.
    # Distribution: ~20% $  |  ~60% $$  |  ~20% $$$
    bucket = abs(hash(restaurant_name.lower().strip())) % 10
    if bucket < 2:
        return "$"
    elif bucket < 8:
        return "$$"
    else:
        return "$$$"


@tool
def pizzas_in_budget(price_tier: str, budget: int) -> str:
    """
    Calculates how many whole pizzas a given budget covers at a specific price tier.

    IMPORTANT: Always call this tool to do the budget calculation.
    Do NOT compute this yourself — use the price tier returned by get_price_tier.

    Args:
        price_tier: The price tier string returned by get_price_tier ('$', '$$', or '$$$').
        budget: Total budget in US dollars (e.g. 100).

    Returns:
        A string such as '4 pizzas (~$22 each, $$ tier) for a $100 budget'.
    """
    _call_tracker["pizzas_in_budget"] = _call_tracker.get("pizzas_in_budget", 0) + 1
    price = _TIER_PRICE.get(price_tier.strip(), _TIER_PRICE["$$"])
    count = budget // price
    return f"{count} pizzas (~${price} each, {price_tier} tier) for a ${budget} budget"


def parse_float_list(env_name: str, default: str) -> list[float]:
    raw = os.getenv(env_name, default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(env_name: str, default: str) -> list[int]:
    raw = os.getenv(env_name, default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def sweep_values() -> dict[str, list[Any]]:
    return {
        "temperature": parse_float_list("TEMP_VALUES", "0.0,0.1,0.3,0.6"),
        "num_ctx": parse_int_list("NUM_CTX_VALUES", "4096,8192,16384"),
        "num_batch": parse_int_list("NUM_BATCH_VALUES", "128,256,512"),
        "num_predict": parse_int_list("NUM_PREDICT_VALUES", "256,512,1024"),
    }


def baseline_options() -> dict[str, Any]:
    return {
        "temperature": BASE_TEMPERATURE,
        "num_ctx": BASE_NUM_CTX,
        "num_batch": BASE_NUM_BATCH,
        "num_predict": BASE_NUM_PREDICT,
    }


def setup_tracer() -> Any:
    """Set up OTEL tracer. Supports two backends, both optional:
    - OTEL_ENABLED=1  → prints spans to console
    - LANGFUSE_ENABLED=1 → sends spans to Langfuse cloud + auto-instruments smolagents
      Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env
    """
    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "0") == "1"
    otel_enabled = os.getenv("OTEL_ENABLED", "0") == "1"

    if not langfuse_enabled and not otel_enabled:
        return None

    try:
        trace = importlib.import_module("opentelemetry.trace")
        sdk_trace = importlib.import_module("opentelemetry.sdk.trace")
        sdk_export = importlib.import_module("opentelemetry.sdk.trace.export")
    except Exception:
        print("OTEL packages not installed — tracing disabled.")
        return None

    provider = sdk_trace.TracerProvider()

    if langfuse_enabled:
        try:
            import base64
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from openinference.instrumentation.smolagents import SmolagentsInstrumentor

            pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            sk = os.getenv("LANGFUSE_SECRET_KEY", "")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            if not pk or not sk:
                print("LANGFUSE_ENABLED=1 but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set.")
            else:
                auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
                exporter = OTLPSpanExporter(
                    endpoint=f"{host}/api/public/otel/v1/traces",
                    headers={"Authorization": f"Basic {auth}"},
                )
                provider.add_span_processor(
                    sdk_export.BatchSpanProcessor(exporter)
                )
                SmolagentsInstrumentor().instrument()
                print(f"Langfuse tracing enabled → {host}")
        except ImportError:
            print("Langfuse packages not installed. Run: pip install opentelemetry-exporter-otlp openinference-instrumentation-smolagents")

    if otel_enabled:
        # Console exporter — prints raw spans to stdout
        provider.add_span_processor(
            sdk_export.BatchSpanProcessor(sdk_export.ConsoleSpanExporter())
        )

    trace.set_tracer_provider(provider)
    return trace.get_tracer("smolsmallagentbase")


def build_model(options: dict[str, Any]) -> LiteLLMModel:
    return LiteLLMModel(
        model_id=OLLAMA_MODEL,
        api_base="http://localhost:11434",
        api_key="ollama",
        timeout=MODEL_TIMEOUT_SECONDS,
        temperature=options["temperature"],
        num_ctx=options["num_ctx"],
        num_batch=options["num_batch"],
        num_predict=options["num_predict"],
    )


def build_agents(model: LiteLLMModel) -> tuple[ToolCallingAgent, CodeAgent]:
    tools = [get_price_tier, pizzas_in_budget, DuckDuckGoSearchTool(), FinalAnswerTool()]
    # 12 steps minimum for the full chain (1 search + 5×get_price_tier + 5×pizzas_in_budget + 1 final_answer)
    # ToolCallingAgent uses one step per tool call so needs headroom above 12
    tool_calling = ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=15,
        verbosity_level=1,
    )
    code_agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        verbosity_level=1,
    )
    return tool_calling, code_agent


def run_agent_once(
    agent_obj: Any,
    agent_label: str,
    tracer: Any,
    param_name: str,
    param_value: Any,
    options: dict[str, Any],
) -> dict[str, Any]:
    global _call_tracker
    _call_tracker = {}  # reset per run so counts are isolated

    start = time.perf_counter()
    success = False
    error_type = ""

    span_cm = tracer.start_as_current_span(f"sweep.{agent_label}") if tracer else None
    if span_cm:
        span_cm.__enter__()

    try:
        agent_obj.run(TASK)
        success = True
    except AgentParsingError as exc:
        error_type = "AgentParsingError"
    except Exception as exc:
        error_type = type(exc).__name__
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        # sum both chained tools: get_price_tier + pizzas_in_budget (5 each = 10 total)
        classify_calls = (
            _call_tracker.get("get_price_tier", 0)
            + _call_tracker.get("pizzas_in_budget", 0)
        )
        # routing_accuracy: fraction of the 10 expected tool calls the agent made
        routing_accuracy = round(min(classify_calls / EXPECTED_CLASSIFY_CALLS, 1.0), 3)

        if span_cm:
            span = importlib.import_module("opentelemetry.trace").get_current_span()
            span.set_attribute("agent.type", agent_label)
            span.set_attribute("sweep.parameter", param_name)
            span.set_attribute("sweep.value", str(param_value))
            span.set_attribute("agent.success", success)
            span.set_attribute("agent.error_type", error_type)
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute("agent.classify_calls", classify_calls)
            span.set_attribute("agent.routing_accuracy", routing_accuracy)
            span_cm.__exit__(None, None, None)

    return {
        "agent": agent_label,
        "parameter": param_name,
        "value": param_value,
        "success": success,
        "duration_ms": round(duration_ms, 2),
        "error_type": error_type,
        "classify_calls": classify_calls,
        "routing_accuracy": routing_accuracy,
        "options": options,
    }


def make_experiment_plan() -> list[dict[str, Any]]:
    values = sweep_values()
    base = baseline_options()
    plan: list[dict[str, Any]] = []

    for param in SWEEP_PARAMS:
        if param not in values:
            continue
        for value in values[param]:
            options = dict(base)
            options[param] = value
            plan.append({
                "parameter": param,
                "value": value,
                "options": options,
            })
    return plan


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec["parameter"], str(rec["value"]), rec["agent"])
        grouped.setdefault(key, []).append(rec)

    summary: list[dict[str, Any]] = []
    for (parameter, value, agent), items in grouped.items():
        latencies = [x["duration_ms"] for x in items]
        spread = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        failures = [x for x in items if not x["success"]]
        routing_scores = [x["routing_accuracy"] for x in items]
        summary.append({
            "parameter": parameter,
            "value": value,
            "agent": agent,
            "runs": len(items),
            "latency_mean_ms": round(statistics.mean(latencies), 2),
            "latency_spread_ms": round(spread, 2),
            "failure_rate": round(len(failures) / len(items), 3),
            "routing_accuracy": round(statistics.mean(routing_scores), 3),
        })

    summary.sort(key=lambda x: (x["parameter"], x["value"], x["agent"]))
    return summary


def build_dashboard(records: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "sweep_results.json"
    html_path = OUTPUT_DIR / "sweep_dashboard.html"

    payload = {
        "model": OLLAMA_MODEL,
        "sweep_runs": SWEEP_RUNS,
        "baseline": baseline_options(),
        "task": TASK,
        "records": records,
        "summary": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    max_mean = max((row["latency_mean_ms"] for row in summary_rows), default=1.0)

    table_rows = "\n".join(
        f"""
        <tr>
          <td>{row['parameter']}</td>
          <td>{row['value']}</td>
          <td><span class="tag {'tc' if row['agent'] == 'tool_calling' else 'ca'}">{row['agent']}</span></td>
          <td>{row['latency_mean_ms']:.0f} ± {row['latency_spread_ms']:.0f} ms
            <div class="track">
              <div class="bar" style="width:{(row['latency_mean_ms'] / max_mean) * 100:.0f}%"></div>
              <div class="spread" style="width:{(row['latency_spread_ms'] / max_mean) * 100:.0f}%"></div>
            </div></td>
          <td>{row['routing_accuracy']:.2f}
            <div class="track"><div class="bar {'good' if row['routing_accuracy'] >= 0.5 else 'bad'}" style="width:{row['routing_accuracy'] * 100:.0f}%"></div></div>
          </td>
          <td>{1 - row['failure_rate']:.2f}
            <div class="track"><div class="bar {'good' if row['failure_rate'] < 0.5 else 'bad'}" style="width:{(1 - row['failure_rate']) * 100:.0f}%"></div></div>
          </td>
        </tr>
        """
        for row in summary_rows
    )

    temp_rows = [r for r in summary_rows if r["parameter"] == "temperature"]
    routing_chart = "\n".join(
        f"""
        <div class="chart-row">
          <span class="chart-label">{r['agent']} T={r['value']}</span>
          <div class="track"><div class="bar {'good' if r['routing_accuracy'] >= 0.5 else 'bad'}" style="width:{r['routing_accuracy'] * 100:.0f}%"></div></div>
          <span class="chart-val">{r['routing_accuracy']:.2f}</span>
        </div>
        """
        for r in temp_rows
    ) or "<p style='color:var(--muted)'>Run with SWEEP_PARAMS=temperature to see routing chart.</p>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Agent Sweep Dashboard</title>
  <style>
    :root{{--bg:#0f172a;--text:#e5e7eb;--muted:#94a3b8;--a:#60a5fa;--b:#a78bfa;}}
    body{{margin:0;background:linear-gradient(180deg,#0b1222,var(--bg));color:var(--text);font-family:ui-sans-serif,system-ui,sans-serif;}}
    .wrap{{max-width:1080px;margin:0 auto;padding:32px 20px 56px;}}
    h1{{margin:0 0 6px;font-size:1.8rem;}} h2{{font-size:1rem;color:var(--muted);margin:0 0 12px;}}
    .sub{{color:var(--muted);margin-bottom:20px;font-size:.88rem;}}
    .card{{background:rgba(17,24,39,.85);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:16px;margin-bottom:14px;}}
    table{{width:100%;border-collapse:collapse;font-size:13px;}}
    th,td{{padding:8px 6px;border-bottom:1px solid rgba(255,255,255,.07);text-align:left;vertical-align:middle;}}
    th{{color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:600;}}
    .track{{position:relative;background:rgba(255,255,255,.08);border-radius:999px;height:8px;overflow:hidden;margin-top:4px;}}
    .bar{{height:100%;background:linear-gradient(90deg,var(--a),var(--b));}}
    .bar.good{{background:linear-gradient(90deg,#22c55e,#16a34a);}}
    .bar.bad{{background:linear-gradient(90deg,#f59e0b,#ef4444);}}
    .spread{{position:absolute;right:0;top:0;bottom:0;background:rgba(245,158,11,.4);}}
    .tag{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;}}
    .tag.tc{{background:rgba(96,165,250,.18);color:#93c5fd;}}
    .tag.ca{{background:rgba(167,139,250,.18);color:#c4b5fd;}}
    .chart-row{{display:grid;grid-template-columns:190px 1fr 48px;gap:10px;align-items:center;margin:8px 0;}}
    .chart-label{{color:var(--muted);font-size:.83rem;}}
    .chart-val{{font-size:.83rem;text-align:right;}}
    .legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:.83rem;color:var(--muted);margin-bottom:14px;}}
    .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle;}}
    code{{background:rgba(255,255,255,.08);padding:2px 6px;border-radius:6px;font-size:.83rem;}}
  </style>
</head>
<body>
<div class="wrap">
  <h1>ToolCallingAgent vs CodeAgent — Routing Experiment</h1>
  <div class="sub">Model: <code>{OLLAMA_MODEL}</code> &middot; Runs/config: <code>{SWEEP_RUNS}</code> &middot; One-variable-at-a-time &middot; Task: find restaurants → get_price_tier → pizzas_in_budget</div>
  <div class="legend">
    <span><span class="dot" style="background:var(--a)"></span>Latency bar</span>
    <span><span class="dot" style="background:rgba(245,158,11,.7)"></span>Latency spread</span>
    <span><span class="dot" style="background:#22c55e"></span>Good routing / success (≥0.5)</span>
    <span><span class="dot" style="background:#ef4444"></span>Poor routing / failures (&lt;0.5)</span>
  </div>
  <section class="card">
    <h2>Latency mean±spread &nbsp;&bull;&nbsp; Tool-routing accuracy &nbsp;&bull;&nbsp; Success rate</h2>
    <table>
      <thead><tr>
        <th>Param</th><th>Value</th><th>Agent</th>
        <th>Latency mean±spread</th>
        <th>Routing accuracy<br><span style="font-weight:400;font-size:10px">(get_price_tier + pizzas_in_budget) / 10</span></th>
        <th>Success rate</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>
  <section class="card">
    <h2>Temperature → tool-routing accuracy per agent</h2>
    <p style="color:var(--muted);font-size:.83rem;margin:0 0 10px">
      1.0 = agent called both <code>get_price_tier</code> and <code>pizzas_in_budget</code> for all 5 restaurants.
      Shows which agent type maintains the full tool chain as temperature increases.
    </p>
    {routing_chart}
  </section>
  <section class="card">
    <p style="color:var(--muted);font-size:.83rem;margin:0">
      Screenshot this page for your post. Raw data: <code>sweep_results.json</code>.
    </p>
  </section>
</div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path, json_path


def _flush(all_records: list[dict[str, Any]], label: str = "checkpoint") -> None:
    """Write dashboard + JSON from whatever records exist so far."""
    if not all_records:
        return
    summary_rows = summarize(all_records)
    html_path, json_path = build_dashboard(all_records, summary_rows)
    print(f"\n[{label}] {len(all_records)} records → {json_path}")


def run_sweep() -> None:
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        try:
            hf_login(token=HF_TOKEN, add_to_git_credential=True)
        except Exception as exc:
            print(f"Hugging Face login skipped: {exc}")

    tracer = setup_tracer()
    experiments = make_experiment_plan()
    if not experiments:
        print("No experiments configured. Check SWEEP_PARAMS.")
        return

    all_records: list[dict[str, Any]] = []
    total_runs = len(experiments) * SWEEP_RUNS
    # Worst-case time estimate: max_steps × MODEL_TIMEOUT_SECONDS × 2 agents per pair
    worst_case_hours = (total_runs * 15 * MODEL_TIMEOUT_SECONDS * 2) / 3600
    print(f"Running {len(experiments)} configs × {SWEEP_RUNS} runs × 2 agents = {total_runs * 2} agent calls")
    print(f"Worst-case wall time: {worst_case_hours:.1f}h  |  typical: {total_runs * 2 * 2 / 60:.0f}–{total_runs * 2 * 4 / 60:.0f} min")
    print(f"Task: {TASK[:100]}...\n")
    print("Results written after every completed run pair. Ctrl+C saves what's done.\n")

    # Graceful Ctrl+C: flush and exit cleanly
    def _sigint_handler(sig: int, frame: Any) -> None:
        print("\n[interrupted] saving partial results...")
        _flush(all_records, "partial")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    completed = 0
    for exp in experiments:
        parameter = exp["parameter"]
        value = exp["value"]
        options = exp["options"]
        print(f"--- {parameter}={value} ---")
        model = build_model(options)
        tool_agent, code_agent = build_agents(model)

        for run_idx in range(SWEEP_RUNS):
            completed += 1
            print(f"  run {run_idx + 1}/{SWEEP_RUNS}  [{completed}/{total_runs}]", end="", flush=True)
            tc = run_agent_once(tool_agent, "tool_calling", tracer, parameter, value, options)
            ca = run_agent_once(code_agent, "code_agent", tracer, parameter, value, options)
            all_records.extend([tc, ca])
            print(
                f"  tc routing={tc['routing_accuracy']:.2f} success={tc['success']} "
                f"| ca routing={ca['routing_accuracy']:.2f} success={ca['success']}"
            )
            # Write after every pair so a kill never loses finished work
            _flush(all_records, f"run {completed}/{total_runs}")

    print("\n=== Sweep complete ===")
    summary_rows = summarize(all_records)
    for row in summary_rows:
        print(
            f"{row['parameter']}={row['value']} | {row['agent']:14s} | "
            f"latency={row['latency_mean_ms']:.0f}±{row['latency_spread_ms']:.0f}ms | "
            f"routing={row['routing_accuracy']:.2f} | "
            f"success={1 - row['failure_rate']:.2f}"
        )


if __name__ == "__main__":
    run_sweep()
