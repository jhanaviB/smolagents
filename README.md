# smolagents experiments

This repo contains some code from my experiements with smolagents, langchain, langgraph and RAG :)

## Projects

### `smolsmallagentbase/`
This is built with [smolagents](https://github.com/huggingface/smolagents) and local Ollama models.

Benchmarks `ToolCallingAgent` vs `CodeAgent` on a chained multi-step pizza planning task. Runs parameter sweeps across temperature, context size, and batch settings, measuring latency, success rate, and tool-routing accuracy. Outputs results to `metrics_output/`.

### `retrievalwebsearch/`
Two experiments using web search with smolagents:
- **`retrievertool.py`** — BM25 retrieval over a local document dict using LangChain, used for speakeasy party planning ideas

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r smolsmallagentbase/requirements.txt
```

Create a `.env` file in the root:
```
HF_TOKEN=your_huggingface_token_here
```

Requires [Ollama](https://ollama.com) running locally with `qwen2.5-coder:14b` pulled:
```bash
ollama pull qwen2.5-coder:14b
```
