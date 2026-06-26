"""
Complaint Copilot — Gradio version (renders inline in Google Colab).

In a Colab cell:
    !pip install gradio sentence-transformers anthropic pandas pyarrow
    # optional: import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
    !python app_gradio.py
or simply run this file's code in a cell. `demo.launch(share=True)` shows it inline
and prints a public link — no tunnel needed.
"""
import os
import re
import numpy as np
import pandas as pd

USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = os.path.join("data", "complaints.parquet")


def _sample_rows():
    base = [
        ("S-0001","Credit reporting","I reported a fraudulent account three times but the bureau keeps saying it is verified and never explains how. This account is not mine."),
        ("S-0002","Credit reporting","A paid collection is still showing as open after 60 days even though I sent the settlement confirmation letter."),
        ("S-0003","Debt collection","No response from the company for over a month after I disputed the amount owed. They keep adding fees that were never explained."),
        ("S-0004","Debt collection","They kept calling me about a debt I already settled. The file was never closed and I get calls multiple times a day."),
        ("S-0005","Mortgage","My mortgage servicer changed my escrow and doubled my payment without any notice and cannot explain the new amount."),
        ("S-0006","Mortgage","The loan officer lost my documents twice and the closing was delayed by weeks, which almost cost me the house."),
        ("S-0007","Money transfer","I sent money to the wrong person through the app and support says nothing can be done. There is no way to reverse the transfer."),
        ("S-0008","Money transfer","The transfer was marked complete but the recipient never received it. It has been a week and the funds are gone."),
        ("S-0009","Credit card","I was charged twice for the same purchase and the billing dispute was denied without any real review."),
        ("S-0010","Credit card","They charged me interest even though I paid the full statement balance before the due date and it keeps compounding."),
        ("S-0011","Student loan","My payments are not being applied to the principal and the servicer keeps putting me in forbearance I never requested."),
        ("S-0012","Checking account","The bank placed a hold on my deposit for ten days with no explanation and I could not pay my rent."),
        ("S-0013","Credit reporting","There are several hard inquiries on my report that I never authorized. Someone may be using my identity."),
        ("S-0014","Money transfer","The exchange rate and hidden fees took almost 15 percent of what I sent abroad and none of it was disclosed."),
    ]
    return pd.DataFrame(base, columns=["complaint_id", "product", "narrative"])


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
        if "narrative" not in df.columns and "complaint_what_happened" in df.columns:
            df = df.rename(columns={"complaint_what_happened": "narrative"})
    else:
        df = _sample_rows()
    return df[df["narrative"].astype(str).str.len() > 0].reset_index(drop=True)


print("Loading data and embedding model...")
df = load_data()
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBED_MODEL)
emb = np.asarray(embedder.encode(df["narrative"].astype(str).tolist(),
                                 normalize_embeddings=True), dtype="float32")
print(f"Ready: {len(df)} complaints | LLM answers: {'on' if USE_LLM else 'off'}")


def llm(prompt, model, max_tokens=400):
    if not USE_LLM:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(model=model, max_tokens=max_tokens,
                                       messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text
    except Exception as e:
        return f"(LLM error: {e})"


def retrieve(query, k=4):
    q = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    scores = emb @ q
    idx = np.argsort(-scores)[:k]
    hits = df.iloc[idx].copy()
    hits["score"] = scores[idx]
    return hits


def respond(query):
    if not query or not query.strip():
        return "Type a question above.", ""
    hits = retrieve(query, k=4)
    context = "\n\n".join(
        f"[{r.complaint_id} | {r.product}] {str(r.narrative)[:500]}" for r in hits.itertuples()
    )
    if USE_LLM:
        prompt = ("You are an analyst answering questions about consumer complaints. "
                  "Use ONLY the complaints below and cite the complaint ids in brackets. "
                  "If unsupported, say so.\n\n"
                  f"Question: {query}\n\nComplaints:\n{context}")
        ans = llm(prompt, ANSWER_MODEL) or "(no answer)"
    else:
        ans = ("Retrieved related complaints (set ANTHROPIC_API_KEY for a synthesized, "
               "cited answer). Top products: "
               + ", ".join(hits["product"].astype(str).tolist()) + ".")
    src = "\n".join(
        f"- **{r.complaint_id}** · {r.product} · score {float(r.score):.2f}  \n  {str(r.narrative)[:220]}"
        for r in hits.itertuples()
    )
    return ans, src


import gradio as gr

with gr.Blocks(title="Complaint Copilot") as demo:
    gr.Markdown("# Complaint Copilot\nAsk in plain English — answers are grounded in real CFPB complaints, with sources.")
    inp = gr.Textbox(label="Ask about complaints",
                     value="Which complaints involve money sent to the wrong person?")
    btn = gr.Button("Ask", variant="primary")
    gr.Examples(["Top credit-reporting issues?",
                 "Who has the worst response times?",
                 "Any emerging payment problems?"], inp)
    out = gr.Markdown(label="Answer")
    src = gr.Markdown(label="Sources")
    btn.click(respond, inp, [out, src])
    inp.submit(respond, inp, [out, src])

if __name__ == "__main__":
    demo.launch(share=True)   # share=True -> inline + public link in Colab
