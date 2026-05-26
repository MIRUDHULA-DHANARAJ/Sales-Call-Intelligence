# 🎯 Sales Call Intelligence Tool v2

> **AI-powered sales call analyzer** built with Hybrid RAG (BM25 + Dense), LangChain, ChromaDB, SQLite, Groq & Whisper.  
> Upload a transcript or audio → 6 AI insights + persistent rep performance tracking.

## 🆕 v2 Improvements
- **Hybrid Search** — BM25 + Dense Vector retrieval fused via Reciprocal Rank Fusion (RRF). Better retrieval quality than pure vector search alone.
- **Rep Performance Tracker** — SQLite persistence across sessions. Track score trends, sentiment history, and coaching patterns per rep over time.

---

## 🚀 Live Demo
👉 **[Try it on HuggingFace Spaces](#)** *(add your link here)*

---

## 📸 What It Does

| Input | Output |
|-------|--------|
| Sales call transcript (.txt) | ⚠️ Objections raised |
| Audio recording (.mp3/.wav) | 🏢 Competitor mentions |
| | 📊 Deal sentiment (Hot/Warm/Cold) |
| | ✅ Next steps + gaps |
| | 🎓 3 coaching tips |
| | 🎯 Deal score (1–10) |

---

## 🏗️ Architecture

```
Input (Audio / Text)
        ↓
[Whisper STT]  ←── runs locally, free
        ↓
[RecursiveCharacterTextSplitter]  chunk_size=600, overlap=80
        ↓
    ┌──────────────────────────────────────┐
    │  HYBRID SEARCH (NEW in v2)           │
    │                                      │
    ├─── [BM25Okapi]   keyword index       │
    └─── [ChromaDB + MiniLM-L6] dense      │
        ↓                                  │
    [Reciprocal Rank Fusion (RRF)]  ───────┘
        fuses both rankings, k=60
        ↓
[Groq: LLaMA3-8b-8192] × 6 tasks
        ↓
[SQLite]  ←── persistent rep tracker (NEW in v2)
        ↓
6 Insights + Rep Performance Dashboard
```

---

## 🛠️ Tech Stack

| Tool | Purpose | Cost |
|------|---------|------|
| **Whisper** | Audio → Text | Free (local) |
| **LangChain** | RAG orchestration | Free |
| **ChromaDB** | Dense vector storage | Free |
| **BM25 (rank-bm25)** | Keyword retrieval | Free |
| **RRF Fusion** | Hybrid ranking | Free (custom code) |
| **HuggingFace Embeddings** | Text → Vectors | Free |
| **Groq API** | LLM inference (LLaMA3) | Free tier |
| **SQLite** | Rep performance persistence | Free (built-in) |
| **Plotly** | Performance charts | Free |
| **Streamlit** | UI | Free |
| **HuggingFace Spaces** | Deployment | Free |

**Total infrastructure cost: $0**

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/sales-call-intelligence
cd sales-call-intelligence
pip install -r requirements.txt
```

### 2. Get Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → Create API Key
3. Copy the key

### 3. Run
```bash
export GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py
```

---

## ☁️ Deploy to HuggingFace Spaces (Free)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as SDK
3. Upload `app.py` and `requirements.txt`
4. Go to **Settings → Secrets** → Add `GROQ_API_KEY`
5. Your app is live in ~2 minutes 🚀

---

## 💡 Design Decisions

**Why Hybrid Search (BM25 + Dense) over pure vector search?**  
Dense vector search is great at semantic similarity but misses exact keyword matches — competitor names like "Salesforce", pricing like "$399", specific objections. BM25 catches these but misses paraphrased meaning. Reciprocal Rank Fusion (RRF) combines both: each retriever ranks all chunks, RRF sums `1/(k+rank)` across both lists. This is exactly how production search systems like Elasticsearch + vector DBs work at scale.

**Why SQLite over a proper database?**  
SQLite is built into Python, needs zero setup, runs free on HuggingFace Spaces, and handles thousands of call records without breaking a sweat. For a portfolio project that needs to demonstrate persistence and analytics, it's the perfect choice. A production system would swap this for PostgreSQL with minimal code changes.

**Why Groq over OpenAI?**  
Groq's free tier delivers 300+ tokens/sec inference. For a demo project, it provides a better live experience with zero cost.

**Why ChromaDB over Pinecone?**  
ChromaDB runs in-memory with no external API. For a Streamlit app where sessions are independent, it's ideal. Pinecone would be better for a persistent multi-user system.

**Why synthetic transcripts?**  
Real sales calls are confidential. Using generated transcripts demonstrates the pipeline while respecting data privacy — exactly what production teams do in staging environments.

---

## 🔮 What I'd Improve in Production

- [x] Hybrid search (BM25 + dense vectors via RRF) ✅ Done
- [x] Rep performance tracking over time (SQLite) ✅ Done
- [ ] Connect to Zoom/Meet API for real-time transcription
- [ ] Fine-tuned classifier for objection detection instead of zero-shot
- [ ] CRM integration (HubSpot free API) to auto-log insights
- [ ] Multi-tenant ChromaDB for persistent per-rep vector stores

---

## 📁 Project Structure

```
sales-call-intelligence/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🤝 Built By

**[Your Name]** — AI/ML Engineer  
[LinkedIn](#) · [GitHub](#) · [HuggingFace](#)

---

*Built to demonstrate end-to-end RAG pipeline engineering with real-world industry application.*
