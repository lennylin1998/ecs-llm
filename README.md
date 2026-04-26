# ecs-llm

Code for the paper **"Evaluating Employee Communication Behaviors Using Large Language Models"**.

The pipeline measures whether frontier LLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Flash)
systematically over- or under-empathize compared to real human support agents, and whether
that miscalibration varies by industry domain (airlines vs. technology).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys before running any LLM steps.

---

## Pipeline Overview

The experiment runs in numbered steps. Each step has a corresponding module in `src/`.

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/pair_builder.py` | Download dataset and extract single-turn complaint–response pairs |

---

## Module Reference

### `src/pair_builder.py` — Step 1: Pair Extraction

Downloads the Customer Support on Twitter dataset via `kagglehub` and reconstructs customer→brand reply threads. Filters to single-turn pairs only — exchanges where the customer tweet is the opening message, not a follow-up — and assigns each pair a domain label (`airline` or `technology`). Output is saved to `data/pairs_raw.csv`.

---

## Data

Large files (`data/`, model responses) are gitignored. Run the pipeline steps in order
to regenerate them. The only external dependency is a Kaggle account for the initial
dataset download (handled automatically by `kagglehub`).
