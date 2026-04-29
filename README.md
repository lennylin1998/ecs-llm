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
| 2 | `src/FIS.ipynb` | Score each complaint with FIS (DistilBERT V_inv + caps / punct / urgency) |
| 2b | `src/FIS.ipynb` — Stratified Sampling section | Bin FIS into tertiles; sample 600 benchmark pairs and 30 pilot pairs |

---

## Module Reference

### `src/pair_builder.py` — Step 1: Pair Extraction

Downloads the Customer Support on Twitter dataset via `kagglehub` and reconstructs customer→brand reply threads. Filters to single-turn pairs only — exchanges where the customer tweet is the opening message, not a follow-up — and assigns each pair a domain label (`airline` or `technology`). Output is saved to `data/pairs_raw.csv`.

---

### `src/FIS.ipynb` — Steps 2 & 2b: FIS Scoring + Stratified Sampling

**FIS Scoring** (`## Remaining FIS Components` section)

Loads `data/output.csv`, which contains pre-computed `v_inv` scores (P(NEGATIVE) from a DistilBERT model fine-tuned on Sentiment140). Computes three remaining components and combines all four into a final FIS score using an 8:0.5:0.5:1 weighting.

| Component | Description |
|-----------|-------------|
| `v_inv` | P(NEGATIVE) from DistilBERT — pre-computed, loaded from `output.csv` |
| `c_ratio` | Proportion of uppercase characters in the tweet |
| `p_dens` | Proportion of `!` and `?` characters |
| `u_flag` | Binary — 1 if any urgency phrase is present (e.g. "worst", "unacceptable") |

- **Input:** `data/output.csv` (must contain `v_inv` column)
- **Output:** `data/output.csv` (adds `c_ratio`, `p_dens`, `u_flag`, `FIS` columns)

**Stratified Sampling** (`## Stratified Sampling` section)

Bins FIS into low / moderate / high tertiles, then draws 100 pairs per cell (2 domains × 3 bins = 600) for the main experiment and 30 additional pairs for LLM-as-judge pilot validation.

- **Input:** `data/output.csv` (must contain `FIS` and `domain` columns)
- **Output:** `data/benchmark_600.csv`, `data/pilot_30.csv`

---

## Data

Large files (`data/`, model weights) are gitignored. The following pre-computed assets
are stored on SharePoint and must be downloaded before running steps that depend on them:

| File / Directory | Required by | SharePoint link |
|---|---|---|
All SharePoint assets are available at this [shared folder](https://uillinoisedu-my.sharepoint.com/:u:/g/personal/zk11_illinois_edu/IQBJernYkTQuQIkNKybrBi--Abw3SWGNNoe1Vz_LrOnHjLQ?e=Pjm42P). Place downloaded files under `ecs-llm/src/`. The raw Twitter CSV is fetched automatically via `kagglehub` and does not need to be downloaded manually.

| File / Directory | Required by | Notes |
|---|---|---|
| `data/pairs_raw.csv` | Step 2 FIS notebook | SharePoint |
| `data/output.csv` | FIS notebook — FIS Scoring section | SharePoint |
| `negative-sentiment-model/` | FIS notebook (`load_trained_model()`) | SharePoint |
| `data/benchmark_600.csv` | Step 4 judge pilot, Step 5 LLM generation | Produced locally by FIS.ipynb |
| `data/pilot_30.csv` | Step 4 judge pilot | Produced locally by FIS.ipynb |
