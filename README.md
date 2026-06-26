# Complaint Copilot — quick-run notebooks

A minimal, runnable scaffold for the **Complaint Copilot** capstone: pull real consumer
complaints from the CFPB public API, turn the free-text narratives into structured fields
with an LLM, embed them for semantic search, and answer natural-language questions grounded
in the retrieved complaints (RAG).

This is a *quick run* meant to prove the pipeline end-to-end on your laptop in a few minutes.
Each stage is intentionally small so it runs without a GPU and, if you want, without any API key.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

(Optional but recommended) set an Anthropic key to enable real LLM extraction and answers:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, the notebooks still run top-to-bottom using transparent fallbacks
(keyword-based extraction and a templated answer), so you can see the whole flow first.

## Run order

1. **`01_ingest_explore.ipynb`** — fetches complaints from the CFPB Open Data API,
   normalizes the fields, does light EDA, and saves `data/complaints.parquet`.
   If the API is unreachable, it falls back to a small built-in synthetic sample so the
   notebook never breaks.
2. **`02_copilot_rag.ipynb`** — loads that parquet (or the sample), runs LLM/heuristic
   extraction, builds embeddings + a cosine-similarity index, and exposes a `answer(query)`
   RAG function with citations, plus a lightweight LLM-as-judge faithfulness check.

## Notes

- **Data source:** CFPB Consumer Complaint Database — https://www.consumerfinance.gov/data-research/consumer-complaints/
  (API docs: https://cfpb.github.io/ccdb5-api/). Government endpoints change occasionally;
  if a request fails, check the current parameters against the docs.
- **Embeddings:** `all-MiniLM-L6-v2` downloads once on first run (needs internet that time).
- **Models:** extraction defaults to `claude-haiku-4-5` (cheap at volume); swap `ANSWER_MODEL`
  to `claude-sonnet-4-6` for richer answers. Anthropic API docs: https://docs.claude.com/en/api/overview
- **The sample data is synthetic** and only there so nothing breaks offline — never present it
  as real CFPB data in your writeup.

## Where this goes next (for the full capstone)

- Scale extraction across the full corpus; store structured fields in DuckDB/Postgres.
- Replace the numpy search with FAISS or Chroma.
- Add BERTopic clustering over the embeddings to surface *emerging* issue clusters.
- Wrap `answer()` in a Streamlit app to build out the dashboard concept.

## Dashboard (`app.py`)

A Streamlit dashboard on top of the same retrieval + RAG functions: a plain-English query
box with grounded, cited answers, summary metrics, and top product/issue charts.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...    # optional; enables synthesized cited answers
streamlit run app.py
```

It loads `data/complaints.parquet` if present (run notebook 01 first for real data),
otherwise the built-in sample, so it always launches.

### Running from Google Colab
Streamlit serves a web app, which Colab can't show inline, so use a tunnel:

```python
!pip install streamlit
!npm install -g localtunnel
!streamlit run app.py &>/content/log.txt &
!npx localtunnel --port 8501
```

Open the printed URL. Simpler alternative for Colab: run the notebooks locally, or wrap
`answer()` in a Gradio `Interface`, which renders inline in a Colab cell with no tunnel.

## Known fix: CFPB API returns 403
The CFPB endpoint (behind Akamai) blocks the default `python-requests` user agent. The fetch
in notebook 01 now sends a normal `User-Agent` header, which resolves it. If you still get a
403, confirm the current parameters against https://cfpb.github.io/ccdb5-api/.

## Gradio version for Colab (`app_gradio.py`)
Streamlit needs a tunnel in Colab; Gradio renders inline. In a cell:

```python
!pip install gradio sentence-transformers anthropic pandas pyarrow
# import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."   # optional
!python app_gradio.py
```

`demo.launch(share=True)` shows the app inline and prints a public link — no tunnel.

## Emerging issue clusters (`03_emerging_clusters.ipynb`)
Clusters narratives into topics over the embeddings and flags which are **emerging**
(growing in the most recent window). Uses BERTopic when installed, otherwise KMeans with
TF-IDF labels, so it always runs. It saves `data/topic_summary.parquet`, and the Streamlit
dashboard automatically shows an **Emerging issue clusters** panel when that file exists.

Run order for the full picture: `01` → `03` → launch `app.py`.
