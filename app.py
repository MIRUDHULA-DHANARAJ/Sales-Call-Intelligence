import os, re, json, sqlite3, tempfile, datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
import whisper

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Call Intelligence Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# ─────────────────────────────────────────────────────────────────────────────
# SAAS ENTERPRISE CLEAN UI THEME (CSS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Workspace Layout resets */
    .stApp {
        background-color: #F8FAFC;
        color: #334155;
    }
    .main .block-container {
        padding: 2rem 3rem !important;
    }
    
    /* Clean, Professional Typography */
    html, body, [class*="css"], p, div {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Enterprise Card Layout System */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .metric-title {
        font-size: 0.775rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1;
    }
    
    /* Input Container Styling */
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    
    /* Clean Action Headers */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    
    /* Hide unneeded Streamlit elements for clean platform appearance */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM SAMPLE DATA
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
# CORE RAG PIPELINE ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, chunks: list[str], embeddings, k: int = 60):
        self.chunks = chunks
        self.k = k
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        docs = [Document(page_content=c, metadata={"index": i}) for i, c in enumerate(chunks)]
        self.vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> str:
        n = len(self.chunks)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranked = np.argsort(bm25_scores)[::-1].tolist()
        dense_docs = self.vectorstore.similarity_search(query, k=min(n, 10))
        dense_ranked = [int(d.metadata["index"]) for d in dense_docs]

        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.k + rank + 1)

        top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return "\n\n".join(self.chunks[i] for i in sorted(top_indices))

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE STRATUM (SQLite persistence layer)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "sales_intelligence.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS call_analyses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                rep_name     TEXT    NOT NULL,
                call_date    TEXT    NOT NULL,
                prospect     TEXT,
                deal_score   INTEGER,
                sentiment    TEXT,
                likelihood   TEXT,
                objections   TEXT,
                competitors  TEXT,
                coaching     TEXT,
                next_steps   TEXT,
                word_count   INTEGER,
                created_at   TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

def save_analysis(rep_name: str, call_date: str, prospect: str, results: dict, word_count: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        score_text = results.get("score", {}).get("answer", "")
        score_num  = extract_score(score_text)
        
        sentiment_text = results.get("sentiment", {}).get("answer", "")
        verdict_line   = [l for l in sentiment_text.split("\n") if "Verdict:" in l]
        sentiment      = verdict_line[0].replace("Verdict:", "").strip() if verdict_line else "Unknown"
        
        likely_line  = [l for l in score_text.split("\n") if "Likelihood:" in l]
        likelihood   = likely_line[0].replace("Likelihood:", "").strip() if likely_line else "Unknown"

        c.execute("""
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

def load_rep_history(rep_name: str = None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        if rep_name and rep_name != "All Reps":
            return pd.read_sql_query("SELECT * FROM call_analyses WHERE rep_name=? ORDER BY call_date DESC", conn, params=(rep_name,))
        return pd.read_sql_query("SELECT * FROM call_analyses ORDER BY call_date DESC", conn)

def get_all_reps() -> list[str]:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT rep_name FROM call_analyses ORDER BY rep_name")
        return [r[0] for r in c.fetchall()]

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND SYSTEM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str:
    return load_whisper_model().transcribe(audio_path)["text"]

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

def build_hybrid_pipeline(transcript: str, groq_api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80, separators=["\n\n", "\n", ".", " "])
    chunks = [doc.page_content for doc in splitter.create_documents([transcript])]
    return HybridRetriever(chunks, get_embeddings()), ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.1, max_tokens=600)

def run_hybrid_analysis(retriever, llm) -> dict:
    tasks = {
        "objections": {"query": "objections concerns prospect", "prompt": "Identify customer objections or items of friction explicitly raised. Provide practical bullet points without pleasantries.", "label": "Customer Objections Friction Analysis"},
        "competitors": {"query": "competitor alternative market options", "prompt": "Extract explicit competitor platforms or internal alternate approaches mentioned by the customer. Outline context.", "label": "Competitive Intelligence Landscape"},
        "sentiment": {"query": "buying intent signals", "prompt": "Assess immediate deal momentum. Format exactly:\nVerdict: [HOT | WARM | COLD]\nReasoning: [Brief overview]", "label": "Deal Velocity Sentiment Evaluation"},
        "next_steps": {"query": "action items follow up tasks", "prompt": "Log concrete milestones assigned or timeline agreements explicitly made, alongside missed opportunities.", "label": "Account Action Items Timeline"},
        "coaching": {"query": "sales representative skills discovery", "prompt": "Act as a senior revenue coach. Provide exactly 3 actionable feedback pillars to improve technical call conversion rates.", "label": "Sales Professional Actionable Coaching Insights"},
        "score": {"query": "deal health pipeline evaluation metrics", "prompt": "Evaluate conversion health. Format exactly:\nScore: [X]/10\nLikelihood: [Low/Medium/High]\nBudget fit: [Yes/No]\nDecision maker reached: [Yes/No]\nKey Factors:\n• [Factor]", "label": "Deal Conversion Probability Diagnostics"},
    }
    from langchain_core.messages import HumanMessage, SystemMessage
    results = {}
    for key, task in tasks.items():
        try:
            context = retriever.retrieve(task["query"], top_k=4)
            response = llm.invoke([SystemMessage(content=task["prompt"]), HumanMessage(content=f"Context:\n{context}\n\nAnalyze.")])
            results[key] = {"answer": response.content.strip(), "label": task["label"]}
        except Exception as e:
            results[key] = {"answer": f"Analysis pipeline timeout: {e}", "label": task["label"]}
    return results

def extract_score(text: str):
    m = re.search(r"Score:\s*(\d+)/10", text, re.IGNORECASE)
    return int(m.group(1)) if m else None

init_db()

# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM CONTROL PANEL (SIDEBAR)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 Core Application Workspace")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq Bearer Token Token", type="password")
        
    st.markdown("---")
    st.markdown("##### Metadata Parameters")
    rep_name = st.text_input("Assigned Sales Professional", placeholder="e.g. Sarah Jenkins")
    prospect_name = st.text_input("Enterprise Account Label", placeholder="e.g. Acme Corp")
    call_date = st.date_input("Inbound Engagement Date")

# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY VIEWPORTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <h2 style='margin:0; padding-bottom:4px;'>Revenue Intelligence Operations Workspace</h2>
        <p style='margin:0; color:#64748B; font-size:0.9rem;'>Continuous speech pipeline pipeline conversion diagnostic workspace v2.4</p>
    </div>
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_tracker = st.tabs(["🚀 Real-time Conversational Audit", "📊 Account Execution Analytics History"])

# ══════════════════════════════════════════════════════════════════════════════
# VIEWPORT 1 — REAL-TIME AUDIT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    # Source selector inside clear workspace card layout
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    c_hdr1, c_hdr2 = st.columns([4, 1])
    with c_hdr1:
        st.markdown("<h4 style='margin:0;'>Active Audio/Transcript Ingestion Vector</h4>", unsafe_allow_html=True)
    with c_hdr2:
        if st.button("Inject System Mock Pipeline", use_container_width=True):
            st.session_state.transcript = SAMPLE_TRANSCRIPT
            st.rerun()
            
    input_type = st.radio("Ingestion Source Form Factor", ["Raw Text Document Payload", "Compressed Audio Stream Ingestion"], horizontal=True, label_visibility="collapsed")
    
    if input_type == "Raw Text Document Payload":
        transcript_input = st.text_area("Payload Window", value=st.session_state.transcript, height=200, placeholder="Insert plaintext raw phone call scripts...", label_visibility="collapsed")
        if transcript_input:
            st.session_state.transcript = transcript_input
    else:
        audio_file = st.file_uploader("Audio Stream File Ingestion", type=["mp3","wav","m4a"])
        if audio_file:
            st.audio(audio_file)
            if st.button("Trigger Pipeline File Transcription", type="primary"):
                with st.spinner("Extracting audio timeline logs via Whisper Engine..."):
                    with tempfile.NamedTemporaryFile(suffix=f".{audio_file.name.split('.')[-1]}", delete=False) as tmp:
                        tmp.write(audio_file.read())
                        t_path = tmp.name
                    st.session_state.transcript = transcribe_audio(t_path)
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Execute Platform Pipeline Audit Run", type="primary", use_container_width=True):
        if not groq_api_key or not st.session_state.transcript.strip():
            st.error("Authentication settings or system transcript buffer payload requirements empty.")
            st.stop()
            
        with st.spinner("Compiling contextual hybrid search queries across Chroma vectors..."):
            retriever, llm = build_hybrid_pipeline(st.session_state.transcript, groq_api_key)
            results = run_hybrid_analysis(retriever, llm)
            
        if rep_name.strip():
            save_analysis(rep_name.strip(), str(call_date), prospect_name.strip() or "General Account", results, len(st.session_state.transcript.split()))

        # Display Pipeline Execution Analytics metrics perfectly inline inside clean structural components
        st.markdown("### Execution Analytics Summary")
        
        score_data  = results.get("score", {})
        score_num   = extract_score(score_data.get("answer", ""))
        sent_text   = results.get("sentiment", {}).get("answer", "")
        v_line      = [l for l in sent_text.split("\n") if "Verdict:" in l]
        verdict     = v_line[0].replace("Verdict:", "").strip() if v_line else "Data Missing"
        
        l_line      = [l for l in score_data.get("answer","").split("\n") if "Likelihood:" in l]
        likelihood  = l_line[0].replace("Likelihood:", "").strip() if l_line else "Data Missing"

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f'<div class="saas-card"><div class="metric-title">Pipeline Health Index Score</div><div class="metric-value" style="color:#4F46E5;">{score_num or "—"}/10</div></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="saas-card"><div class="metric-title">Customer Urgency Classification</div><div class="metric-value">{verdict}</div></div>', unsafe_allow_html=True)
        with m_col3:
            st.markdown(f'<div class="saas-card"><div class="metric-title">Conversion Closure Probability</div><div class="metric-value">{likelihood}</div></div>', unsafe_allow_html=True)

        st.markdown("### Comprehensive Platform Structural Findings Output")
        for key in ["objections", "competitors", "sentiment", "next_steps", "coaching", "score"]:
            if key not in results: continue
            d = results[key]
            st.markdown(f'<div class="saas-card">🔬 <b style="color:#0F172A; font-size:1rem;">{d["label"]}</b><div style="margin-top:12px; font-size:0.925rem; line-height:1.6; color:#334155;">{d["answer"]}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEWPORT 2 — ENTERPRISE REVENUE TRACKER LOGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_tracker:
    all_reps = get_all_reps()
    if not all_reps:
        st.info("System operational warehouse database logs dry. Execute initial platform analysis loops above to persist run records.")
    else:
        f_col1, f_col2 = st.columns([3, 1])
        with f_col1:
            selected_rep = st.selectbox("Active Corporate Professional Workspace Node Filter", ["All Reps"] + all_reps, label_visibility="collapsed")
        with f_col2:
            if st.button("Synchronize Logs", use_container_width=True): st.rerun()

        df = load_rep_history(selected_rep)
        if not df.empty:
            # Operational High Level Executive KPIs
            k1, k2, k3, k4 = st.columns(4)
            valid_scores = df["deal_score"].dropna()
            avg_s = round(valid_scores.mean(), 1) if len(valid_scores) else "—"
            peak_s = int(valid_scores.max()) if len(valid_scores) else "—"
            hot_deals = round((df["sentiment"].str.contains("HOT", case=False, na=False).sum() / len(df)) * 100)

            with k1: st.markdown(f'<div class="saas-card"><div class="metric-title">Cohort Mean Score</div><div class="metric-value">{avg_s}/10</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="saas-card"><div class="metric-title">Peak Execution Ceiling</div><div class="metric-value">{peak_s}/10</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="saas-card"><div class="metric-title">Aggregated Account Runs</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
            with k4: st.markdown(f'<div class="saas-card"><div class="metric-title">High Velocity Opportunity Mix</div><div class="metric-value">{hot_deals}%</div></div>', unsafe_allow_html=True)

            # Data Charts Set clean background properties explicitly
            trend_df = df[df["deal_score"].notna()].copy()
            trend_df["call_date"] = pd.to_datetime(trend_df["call_date"])
            trend_df = trend_df.sort_values("call_date")

            if len(trend_df) >= 2:
                st.markdown("#### Dynamic Revenue Health Performance Vector Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend_df["call_date"], y=trend_df["deal_score"], mode="lines+markers", line=dict(color="#4F46E5", width=3), marker=dict(size=8)))
                fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", font=dict(color="#334155"), yaxis=dict(range=[0, 11]), height=280, margin=dict(l=40, r=20, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Enterprise Execution Account Auditing Ledger Logs")
            st.dataframe(df[["call_date", "rep_name", "prospect", "deal_score", "sentiment", "likelihood"]].rename(columns={"call_date": "Date", "rep_name": "Representative", "prospect": "Account", "deal_score": "Score Index", "sentiment": "Velocity Profile", "likelihood": "Conversion Threshold"}), use_container_width=True, hide_index=True)