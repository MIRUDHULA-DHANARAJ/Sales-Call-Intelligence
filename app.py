import os
import re
import json
import sqlite3
import tempfile
import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi
import whisper

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Call Intelligence",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN UI — WHITE BG, BRIGHT COLOURED TEXT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: #FFFFFF !important;
        color: #1a1a2e !important;
    }

    .stApp {
        background-color: #FFFFFF;
    }

    .main .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1200px;
    }

    /* ── Page title ── */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #6C3AE8;          /* vivid violet */
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    .page-sub {
        font-size: 0.9rem;
        color: #0EA5E9;          /* sky blue */
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #F97316;          /* orange */
        margin-bottom: 6px;
    }

    /* ── Cards ── */
    .card {
        background: #FFFFFF;
        border: 1.5px solid #E4E4F0;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(108, 58, 232, 0.06);
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #FAFAFA;
        border: 1.5px solid #E4E4F0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    .mv-violet  { color: #6C3AE8; }
    .mv-green   { color: #10B981; }
    .mv-orange  { color: #F97316; }
    .mv-blue    { color: #0EA5E9; }

    /* ── Result block headings ── */
    .result-heading {
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .rh-violet  { color: #6C3AE8; }
    .rh-teal    { color: #0D9488; }
    .rh-pink    { color: #EC4899; }
    .rh-amber   { color: #D97706; }
    .rh-blue    { color: #2563EB; }
    .rh-green   { color: #059669; }

    .result-body {
        font-size: 0.9rem;
        color: #374151;
        line-height: 1.65;
        white-space: pre-wrap;
        font-family: 'DM Mono', monospace;
    }

    /* Streamlit native component tweaks */
    .stTextArea textarea, .stTextInput input {
        border: 1.5px solid #C7D2FE !important;
        border-radius: 10px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.875rem !important;
    }
    .stButton > button[kind="primary"] {
        background: #6C3AE8 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #6C3AE8 !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE TRANSCRIPT
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_TRANSCRIPT = """
Sales Rep (Priya): Hi Alex, thanks for taking the time today. How's the quarter going?
Prospect (Alex): Honestly rough. We're drowning in manual reporting — 12 hours a week just pulling data from different tools and dumping it into spreadsheets.
Priya: Are you currently using any tool to automate that?
Alex: We tried Salesforce's built-in reporting but it's terrible. Way too rigid. We even looked at Tableau but the pricing was insane for what we need.
Priya: What does your ideal solution look like?
Alex: Connects to our CRM, pulls data automatically, drag-and-drop dashboards, no SQL required. Budget cap is $500 a month.
Priya: Our Growth plan is $399/month for up to 10 seats. Native CRM integrations, no SQL required.
Alex: Interesting. But we've been burned before — signed a contract last year, 4-month implementation, half the features didn't work.
Priya: We offer a 30-day pilot, no contract, no credit card. Most teams are live in a week.
Alex: A week sounds aggressive. Our IT team is backed up.
Priya: We handle 90% of setup. IT involvement under 2 hours. Done this with Freshworks and Zoho.
Alex: I'd need to loop in my manager. She's very particular about new tools.
Priya: I can put together a one-pager — ROI, security, implementation timeline.
Alex: Yeah, send it by Thursday.
Priya: I'll have it Wednesday. Should we schedule a call with her next week?
Alex: Let's see how she responds to the doc first.
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "sales_intelligence.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                rep_name    TEXT NOT NULL,
                call_date   TEXT NOT NULL,
                prospect    TEXT,
                deal_score  INTEGER,
                sentiment   TEXT,
                likelihood  TEXT,
                objections  TEXT,
                competitors TEXT,
                coaching    TEXT,
                next_steps  TEXT,
                word_count  INTEGER,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

def save_analysis(rep_name, call_date, prospect, results, word_count):
    score_text     = results.get("score", {}).get("answer", "")
    score_num      = extract_score(score_text)
    sentiment_text = results.get("sentiment", {}).get("answer", "")
    verdict_line   = [l for l in sentiment_text.split("\n") if "Verdict:" in l]
    sentiment      = verdict_line[0].replace("Verdict:", "").strip() if verdict_line else "Unknown"
    likely_line    = [l for l in score_text.split("\n") if "Likelihood:" in l]
    likelihood     = likely_line[0].replace("Likelihood:", "").strip() if likely_line else "Unknown"

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO call_analyses
                (rep_name, call_date, prospect, deal_score, sentiment,
                 likelihood, objections, competitors, coaching, next_steps, word_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rep_name, call_date, prospect, score_num, sentiment, likelihood,
            results.get("objections", {}).get("answer", ""),
            results.get("competitors", {}).get("answer", ""),
            results.get("coaching",    {}).get("answer", ""),
            results.get("next_steps",  {}).get("answer", ""),
            word_count
        ))
        conn.commit()

def load_history(rep_name=None):
    with sqlite3.connect(DB_PATH) as conn:
        if rep_name and rep_name != "All Reps":
            return pd.read_sql_query(
                "SELECT * FROM call_analyses WHERE rep_name=? ORDER BY call_date DESC",
                conn, params=(rep_name,)
            )
        return pd.read_sql_query(
            "SELECT * FROM call_analyses ORDER BY call_date DESC", conn
        )

def get_all_reps():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT rep_name FROM call_analyses ORDER BY rep_name"
        ).fetchall()
        return [r[0] for r in rows]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def extract_score(text: str):
    m = re.search(r"Score:\s*(\d+)/10", text, re.IGNORECASE)
    return int(m.group(1)) if m else None

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

def transcribe_audio(path: str) -> str:
    return load_whisper().transcribe(path)["text"]

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RAG RETRIEVER  (BM25 + Dense, fused via RRF)
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, chunks: list, embeddings, k: int = 60):
        self.chunks = chunks
        self.k = k
        tokenized  = [c.lower().split() for c in chunks]
        self.bm25  = BM25Okapi(tokenized)
        docs = [Document(page_content=c, metadata={"index": i}) for i, c in enumerate(chunks)]
        self.vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    def retrieve(self, query: str, top_k: int = 4) -> str:
        n = len(self.chunks)

        # BM25 ranking
        bm25_scores  = self.bm25.get_scores(query.lower().split())
        bm25_ranked  = np.argsort(bm25_scores)[::-1].tolist()

        # Dense ranking
        dense_docs   = self.vectorstore.similarity_search(query, k=min(n, 10))
        dense_ranked = [int(d.metadata["index"]) for d in dense_docs]

        # Reciprocal Rank Fusion
        rrf = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.k + rank + 1)

        top_indices = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        return "\n\n".join(self.chunks[i] for i in sorted(top_indices))

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILDER & ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(transcript: str, api_key: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks    = [d.page_content for d in splitter.create_documents([transcript])]
    retriever = HybridRetriever(chunks, get_embeddings())
    llm       = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key,
                         temperature=0.1, max_tokens=600)
    return retriever, llm

ANALYSIS_TASKS = {
    "objections": {
        "query":  "objections concerns friction prospect",
        "prompt": "List every customer objection or concern raised in this call. Use clear bullet points.",
        "label":  "Customer Objections"
    },
    "competitors": {
        "query":  "competitor alternative tools mentioned",
        "prompt": "Extract any competitor tools or alternatives the prospect mentioned. Give brief context for each.",
        "label":  "Competitors Mentioned"
    },
    "sentiment": {
        "query":  "buying intent signals urgency",
        "prompt": "Assess deal momentum. Respond in this exact format:\nVerdict: [HOT | WARM | COLD]\nReasoning: [2-3 sentences]",
        "label":  "Deal Sentiment"
    },
    "next_steps": {
        "query":  "action items follow up tasks timeline",
        "prompt": "List concrete next steps agreed upon. Note any missed follow-up opportunities.",
        "label":  "Next Steps & Actions"
    },
    "coaching": {
        "query":  "sales rep skills discovery technique",
        "prompt": "You are a senior sales coach. Give exactly 3 specific, actionable coaching tips to improve this rep's performance.",
        "label":  "Coaching Feedback"
    },
    "score": {
        "query":  "deal health evaluation conversion probability",
        "prompt": (
            "Score this deal. Use this exact format:\n"
            "Score: [X]/10\n"
            "Likelihood: [Low | Medium | High]\n"
            "Budget fit: [Yes | No]\n"
            "Decision maker reached: [Yes | No]\n"
            "Key Factors:\n• [factor 1]\n• [factor 2]\n• [factor 3]"
        ),
        "label":  "Deal Score & Probability"
    },
}

def run_analysis(retriever, llm) -> dict:
    results = {}
    for key, task in ANALYSIS_TASKS.items():
        try:
            context  = retriever.retrieve(task["query"])
            response = llm.invoke([
                SystemMessage(content=task["prompt"]),
                HumanMessage(content=f"Transcript excerpt:\n{context}\n\nAnalyze.")
            ])
            results[key] = {"answer": response.content.strip(), "label": task["label"]}
        except Exception as e:
            results[key] = {"answer": f"Error: {e}", "label": task["label"]}
    return results

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP FOR RESULT SECTIONS
# ─────────────────────────────────────────────────────────────────────────────
HEADING_COLOURS = {
    "objections": ("rh-violet", "#6C3AE8"),
    "competitors": ("rh-teal",  "#0D9488"),
    "sentiment":   ("rh-pink",  "#EC4899"),
    "next_steps":  ("rh-amber", "#D97706"),
    "coaching":    ("rh-blue",  "#2563EB"),
    "score":       ("rh-green", "#059669"),
}

# ─────────────────────────────────────────────────────────────────────────────
# INIT DB
# ─────────────────────────────────────────────────────────────────────────────
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📞 Sales Intelligence")
    st.markdown("---")

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("##### Call Details")
    rep_name      = st.text_input("Sales Rep Name",  placeholder="e.g. Priya Sharma")
    prospect_name = st.text_input("Prospect / Account", placeholder="e.g. Acme Corp")
    call_date     = st.date_input("Call Date")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📞 Sales Call Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Analyse call transcripts · Score deals · Track reps over time</div>', unsafe_allow_html=True)

tab_analyse, tab_history = st.tabs(["🔍 Analyse a Call", "📊 Rep History"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
with tab_analyse:

    st.markdown('<div class="section-label">Transcript Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input type",
        ["Paste Text", "Upload Audio"],
        horizontal=True,
        label_visibility="collapsed"
    )

    col_input, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Load Sample", use_container_width=True):
            st.session_state.transcript = SAMPLE_TRANSCRIPT
            st.rerun()

    if input_mode == "Paste Text":
        transcript_text = st.text_area(
            "Transcript",
            value=st.session_state.transcript,
            height=220,
            placeholder="Paste your call transcript here…",
            label_visibility="collapsed"
        )
        if transcript_text:
            st.session_state.transcript = transcript_text

    else:
        audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])
        if audio_file:
            st.audio(audio_file)
            if st.button("Transcribe Audio", type="primary"):
                with st.spinner("Transcribing with Whisper…"):
                    with tempfile.NamedTemporaryFile(
                        suffix=f".{audio_file.name.split('.')[-1]}", delete=False
                    ) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    st.session_state.transcript = transcribe_audio(tmp_path)
                    st.rerun()

    st.markdown("---")

    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()
        if not st.session_state.transcript.strip():
            st.error("Please enter or load a transcript first.")
            st.stop()

        with st.spinner("Running hybrid RAG analysis…"):
            retriever, llm = build_pipeline(st.session_state.transcript, groq_api_key)
            results = run_analysis(retriever, llm)

        # Save if rep name provided
        if rep_name.strip():
            save_analysis(
                rep_name.strip(),
                str(call_date),
                prospect_name.strip() or "Unknown",
                results,
                len(st.session_state.transcript.split())
            )

        # ── Summary metrics ──
        st.markdown('<div class="section-label" style="margin-top:1rem;">Summary</div>', unsafe_allow_html=True)

        score_text  = results.get("score", {}).get("answer", "")
        score_num   = extract_score(score_text)
        sent_text   = results.get("sentiment", {}).get("answer", "")
        verdict_ln  = [l for l in sent_text.split("\n") if "Verdict:" in l]
        verdict     = verdict_ln[0].replace("Verdict:", "").strip() if verdict_ln else "—"
        likely_ln   = [l for l in score_text.split("\n") if "Likelihood:" in l]
        likelihood  = likely_ln[0].replace("Likelihood:", "").strip() if likely_ln else "—"

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Deal Score</div>
                <div class="metric-value mv-violet">{score_num or "—"}/10</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            color_cls = "mv-green" if "HOT" in verdict.upper() else ("mv-orange" if "WARM" in verdict.upper() else "mv-blue")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Deal Sentiment</div>
                <div class="metric-value {color_cls}">{verdict}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Close Likelihood</div>
                <div class="metric-value mv-orange">{likelihood}</div>
            </div>""", unsafe_allow_html=True)

        # ── Detailed results ──
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Detailed Analysis</div>', unsafe_allow_html=True)

        for key in ["objections", "competitors", "sentiment", "next_steps", "coaching", "score"]:
            if key not in results:
                continue
            d        = results[key]
            cls, hex_col = HEADING_COLOURS[key]
            st.markdown(f"""
            <div class="card">
                <div class="result-heading {cls}">{d['label']}</div>
                <div class="result-body">{d['answer']}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    all_reps = get_all_reps()

    if not all_reps:
        st.info("No analyses saved yet. Run an analysis with a rep name to start tracking history.")
    else:
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            selected_rep = st.selectbox(
                "Filter by rep",
                ["All Reps"] + all_reps,
                label_visibility="collapsed"
            )
        with col_f2:
            if st.button("Refresh", use_container_width=True):
                st.rerun()

        df = load_history(selected_rep)

        if not df.empty:
            # KPI row
            valid_scores = df["deal_score"].dropna()
            avg_s   = round(valid_scores.mean(), 1) if len(valid_scores) else "—"
            peak_s  = int(valid_scores.max())        if len(valid_scores) else "—"
            hot_pct = round(
                df["sentiment"].str.contains("HOT", case=False, na=False).sum()
                / len(df) * 100
            )

            k1, k2, k3, k4 = st.columns(4)
            kpi_data = [
                ("Avg Score",   f"{avg_s}/10", "mv-violet"),
                ("Peak Score",  f"{peak_s}/10", "mv-green"),
                ("Total Calls", str(len(df)), "mv-blue"),
                ("Hot Deals",   f"{hot_pct}%", "mv-orange"),
            ]
            for col, (label, value, cls) in zip([k1, k2, k3, k4], kpi_data):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value {cls}">{value}</div>
                    </div>""", unsafe_allow_html=True)

            # Trend chart
            trend_df = df[df["deal_score"].notna()].copy()
            trend_df["call_date"] = pd.to_datetime(trend_df["call_date"])
            trend_df = trend_df.sort_values("call_date")

            if len(trend_df) >= 2:
                st.markdown('<div class="section-label" style="margin-top:1.2rem;">Score Trend</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_df["call_date"],
                    y=trend_df["deal_score"],
                    mode="lines+markers",
                    line=dict(color="#6C3AE8", width=3),
                    marker=dict(size=8, color="#EC4899")
                ))
                fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                    font=dict(family="DM Sans", color="#334155"),
                    yaxis=dict(range=[0, 11], title="Score"),
                    xaxis=dict(title="Date"),
                    height=260,
                    margin=dict(l=40, r=20, t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            # History table
            st.markdown('<div class="section-label" style="margin-top:0.8rem;">Call Log</div>', unsafe_allow_html=True)
            st.dataframe(
                df[["call_date", "rep_name", "prospect", "deal_score", "sentiment", "likelihood"]]
                .rename(columns={
                    "call_date":  "Date",
                    "rep_name":   "Rep",
                    "prospect":   "Account",
                    "deal_score": "Score",
                    "sentiment":  "Sentiment",
                    "likelihood": "Likelihood"
                }),
                use_container_width=True,
                hide_index=True
            )