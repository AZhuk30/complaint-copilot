"""
Complaint Copilot — Streamlit dashboard.

Run locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...        # optional; enables LLM answers
    streamlit run app.py

It loads data/complaints.parquet (from notebook 01) if present, otherwise a small
built-in sample, so it runs even with no data and no API key.
"""
import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = os.path.join("data", "complaints.parquet")


def _sample_rows():
    return [
        {"complaint_id":"S-0001","product":"Credit reporting","issue":"Incorrect information on your report","company":"Acme Bureau","state":"WA","timely":"Yes","complaint_what_happened":"I reported a fraudulent account three times but the bureau keeps saying it is verified and never explains how. This account is not mine and is hurting my score."},
        {"complaint_id":"S-0002","product":"Credit reporting","issue":"Incorrect information on your report","company":"Acme Bureau","state":"CA","timely":"Yes","complaint_what_happened":"A paid collection is still showing as open after 60 days even though I sent the confirmation letter that it was settled in full."},
        {"complaint_id":"S-0003","product":"Debt collection","issue":"Attempts to collect debt not owed","company":"Recovery Partners","state":"TX","timely":"No","complaint_what_happened":"There was no response from the company for over a month after I disputed the amount owed. They keep adding fees that were never explained."},
        {"complaint_id":"S-0004","product":"Debt collection","issue":"Communication tactics","company":"Recovery Partners","state":"FL","timely":"Yes","complaint_what_happened":"They kept calling me about a debt I already settled. The file was never closed and I get calls multiple times a day."},
        {"complaint_id":"S-0005","product":"Mortgage","issue":"Trouble during payment process","company":"HomeServe Bank","state":"WA","timely":"Yes","complaint_what_happened":"My mortgage servicer changed my escrow and doubled my payment without any notice. When I call, no one can explain the new amount."},
        {"complaint_id":"S-0006","product":"Mortgage","issue":"Applying for a mortgage","company":"HomeServe Bank","state":"OR","timely":"Yes","complaint_what_happened":"The loan officer lost my documents twice and the closing was delayed by weeks, which almost cost me the house."},
        {"complaint_id":"S-0007","product":"Money transfer","issue":"Fraud or scam","company":"PayQuick","state":"NY","timely":"Yes","complaint_what_happened":"I sent money to the wrong person through the app and support says nothing can be done to recover it. There is no way to reverse the transfer."},
        {"complaint_id":"S-0008","product":"Money transfer","issue":"Other transaction problem","company":"PayQuick","state":"GA","timely":"Yes","complaint_what_happened":"The transfer was marked complete but the recipient never received it. It has been a week and the funds are just gone."},
        {"complaint_id":"S-0009","product":"Credit card","issue":"Problem with a purchase shown on statement","company":"Metro Card","state":"IL","timely":"Yes","complaint_what_happened":"I was charged twice for the same purchase and the billing dispute was denied without any real review of the evidence I uploaded."},
        {"complaint_id":"S-0010","product":"Credit card","issue":"Fees or interest","company":"Metro Card","state":"WA","timely":"Yes","complaint_what_happened":"They charged me interest even though I paid the full statement balance before the due date. The interest keeps compounding."},
        {"complaint_id":"S-0011","product":"Student loan","issue":"Dealing with your lender or servicer","company":"EduServ","state":"PA","timely":"Yes","complaint_what_happened":"My payments are not being applied to the principal and the servicer keeps putting me in forbearance that I never requested."},
        {"complaint_id":"S-0012","product":"Checking account","issue":"Managing an account","company":"First River Bank","state":"WA","timely":"No","complaint_what_happened":"The bank placed a hold on my deposit for ten days with no explanation and I could not pay my rent because of it."},
        {"complaint_id":"S-0013","product":"Credit reporting","issue":"Improper use of your report","company":"Acme Bureau","state":"NV","timely":"Yes","complaint_what_happened":"There are several hard inquiries on my report that I never authorized. I think someone is using my identity to open accounts."},
        {"complaint_id":"S-0014","product":"Money transfer","issue":"Other transaction problem","company":"PayQuick","state":"CA","timely":"Yes","complaint_what_happened":"The exchange rate and hidden fees took almost 15 percent of what I sent abroad and none of it was disclosed up front."},
    ]


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
        source = "data/complaints.parquet"
    else:
        df = pd.DataFrame(_sample_rows()).rename(columns={"complaint_what_happened": "narrative"})
        source = "built-in sample"
    if "narrative" not in df.columns and "complaint_what_happened" in df.columns:
        df = df.rename(columns={"complaint_what_happened": "narrative"})
    df = df[df["narrative"].astype(str).str.len() > 0].reset_index(drop=True)
    return df, source


@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_data(show_spinner=False)
def get_embeddings(narratives):
    embedder = get_embedder()
    emb = embedder.encode(list(narratives), normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def llm(prompt, model, max_tokens=400):
    if not USE_LLM:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as e:
        st.warning(f"LLM call failed: {e}")
        return None


def retrieve(df, emb, query, k=4):
    embedder = get_embedder()
    q = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    scores = emb @ q
    idx = np.argsort(-scores)[:k]
    hits = df.iloc[idx].copy()
    hits["score"] = scores[idx]
    return hits


def answer(df, emb, query, k=4):
    hits = retrieve(df, emb, query, k=k)
    context = "\n\n".join(
        f"[{r.complaint_id} | {r.product}] {str(r.narrative)[:500]}"
        for r in hits.itertuples()
    )
    if USE_LLM:
        prompt = (
            "You are an analyst answering questions about consumer complaints. "
            "Use ONLY the complaints below. Cite the complaint ids you used in brackets. "
            "If the complaints do not support an answer, say so.\n\n"
            f"Question: {query}\n\nComplaints:\n{context}"
        )
        out = llm(prompt, ANSWER_MODEL)
        if out:
            return out, hits
    summary = (
        f"Retrieved {len(hits)} related complaints. Top matching products: "
        + ", ".join(hits["product"].astype(str).tolist())
        + ". (Set ANTHROPIC_API_KEY for a synthesized, cited answer.)"
    )
    return summary, hits


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Complaint Copilot", layout="wide")

st.markdown(
    """
    <style>
      .cc-badge {background:#e7f6ec;color:#1a7f37;font-size:13px;padding:3px 10px;border-radius:8px;}
      .cc-source {background:#f6f7f9;border:1px solid #e6e8eb;border-radius:8px;padding:8px 12px;margin-bottom:6px;}
      .cc-source small {color:#6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

df, source = load_data()
emb = get_embeddings(tuple(df["narrative"].astype(str)))

with st.sidebar:
    st.subheader("Status")
    st.write("**Data source:**", source)
    st.write("**Complaints loaded:**", len(df))
    st.write("**LLM answers:**", "on" if USE_LLM else "off (no API key)")
    st.write("**Answer model:**", ANSWER_MODEL if USE_LLM else "—")
    st.caption("Run notebook 01 to replace the sample with live CFPB data.")

c1, c2 = st.columns([4, 1])
with c1:
    st.title("Complaint Copilot")
    st.caption("CFPB complaint intelligence — ask in plain English, answers grounded in real complaints")
with c2:
    st.markdown('<span class="cc-badge">● Live · daily</span>', unsafe_allow_html=True)

st.divider()

if "query" not in st.session_state:
    st.session_state.query = "Which complaints involve money sent to the wrong person?"

examples = [
    "Top credit-reporting issues?",
    "Who has the worst response times?",
    "Any emerging payment problems?",
]
ec = st.columns(len(examples))
for i, ex in enumerate(examples):
    if ec[i].button(ex, use_container_width=True):
        st.session_state.query = ex

q = st.text_input("Ask about complaints", value=st.session_state.query, key="query")

if q:
    ans, hits = answer(df, emb, q, k=4)
    st.markdown("#### Answer")
    st.write(ans)
    st.markdown("**Sources**")
    for r in hits.itertuples():
        st.markdown(
            f'<div class="cc-source"><small>{r.complaint_id} · {r.product} '
            f'· score {float(r.score):.2f}</small><br>{str(r.narrative)[:300]}</div>',
            unsafe_allow_html=True,
        )
    if USE_LLM:
        st.caption("Grounded in the complaints above · extraction can be checked with an LLM-judge")

st.divider()

m = st.columns(4)
m[0].metric("Complaints", f"{len(df):,}")
m[1].metric("Products", df["product"].nunique())
m[2].metric("Companies", df["company"].nunique() if "company" in df.columns else "—")
if "timely" in df.columns:
    timely = (df["timely"].astype(str).str.lower() == "yes").mean()
    m[3].metric("Timely response", f"{timely*100:.0f}%")
else:
    m[3].metric("Timely response", "—")

st.divider()

left, right = st.columns(2)
with left:
    st.markdown("#### Top products")
    st.bar_chart(df["product"].value_counts().head(8))
with right:
    st.markdown("#### Top issues")
    if "issue" in df.columns:
        st.bar_chart(df["issue"].value_counts().head(8))
    else:
        st.info("No 'issue' column in this dataset.")

TOPIC_PATH = os.path.join("data", "topic_summary.parquet")
if os.path.exists(TOPIC_PATH):
    st.divider()
    st.markdown("#### Emerging issue clusters")
    st.caption("Topic clusters ranked by recent-vs-prior growth. Run notebook 03 to refresh.")
    ts = pd.read_parquet(TOPIC_PATH)
    cols = [c for c in ["label", "recent", "prior", "size", "growth"] if c in ts.columns]
    rising = ts.sort_values("growth", ascending=False).head(8)[cols]
    st.dataframe(rising, hide_index=True, use_container_width=True)
    if "growth" in ts.columns and "label" in ts.columns:
        chart = ts.sort_values("growth", ascending=False).head(8).set_index("label")["growth"]
        st.bar_chart(chart)
else:
    st.divider()
    st.info("Run notebook 03 (Emerging Issue Clusters) to add a cluster panel here.")
