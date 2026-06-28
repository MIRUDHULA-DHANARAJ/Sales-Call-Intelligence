# 📞 Sales Call Intelligence Platform

A Streamlit app that analyses sales call transcripts using a **hybrid RAG pipeline** (BM25 + dense vector search). Paste a transcript or upload an audio file and get instant AI-powered insights — objections, competitor mentions, deal score, coaching feedback, and more.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ Audio transcription | Upload MP3/WAV/M4A — transcribed via OpenAI Whisper |
| 🔍 Hybrid retrieval | BM25 + ChromaDB vectors fused with Reciprocal Rank Fusion |
| 🤖 LLM analysis | Powered by Groq (Llama 3.1 8B Instant) |
| 📊 Deal scoring | Score/10, close likelihood, budget fit, decision-maker reached |
| 🧑‍🏫 Coaching insights | 3 actionable feedback points per call |
| 🗂️ Rep history | SQLite persistence — track scores and sentiment over time |
| 🎨 Clean UI | White background, bright colour-coded sections |

---



## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/sales-call-intelligence.git
cd sales-call-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

Either export it as an environment variable:

```bash
export GROQ_API_KEY=gsk_...
```

Or enter it directly in the sidebar when the app is running.  
Get a free key at [console.groq.com](https://console.groq.com).

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
plotly
groq
langchain
langchain-community
langchain-groq
langchain-core
langchain-text-splitters
chromadb
sentence-transformers
rank-bm25
numpy
openai-whisper
```

Save the above as `requirements.txt`, or install manually:

```bash
pip install streamlit pandas plotly groq langchain langchain-community \
            langchain-groq langchain-core langchain-text-splitters \
            chromadb sentence-transformers rank-bm25 numpy openai-whisper
```

> **Note:** Whisper requires `ffmpeg` for audio processing.  
> Install it via `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux).

---

## 🏗️ Architecture

```
app.py
│
├── HybridRetriever          # BM25 + ChromaDB + RRF fusion
├── build_pipeline()         # Chunks transcript → builds retriever + Groq LLM
├── run_analysis()           # Runs 6 analysis tasks via RAG
│
├── SQLite (sales_intelligence.db)
│   └── call_analyses table  # Persists every run per rep
│
└── Streamlit UI
    ├── Tab 1: Analyse a Call
    └── Tab 2: Rep History + Score Trend Chart
```

### Analysis tasks

| Key | What it extracts |
|---|---|
| `objections` | Customer friction points and concerns |
| `competitors` | Tools or alternatives the prospect mentioned |
| `sentiment` | HOT / WARM / COLD deal momentum |
| `next_steps` | Agreed action items and follow-ups |
| `coaching` | 3 specific coaching tips for the rep |
| `score` | Score/10, likelihood, budget fit, decision-maker |

---

## ⚙️ Configuration

All configurable values are at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `sales_intelligence.db` | SQLite database path |
| `chunk_size` | `600` | Transcript chunk size for RAG |
| `chunk_overlap` | `80` | Overlap between chunks |
| `model_name` | `llama-3.1-8b-instant` | Groq model |
| `max_tokens` | `600` | Max tokens per LLM response |
| Whisper model | `base` | Change to `small`/`medium` for better accuracy |

---

## 📁 Project Structure

```
sales-call-intelligence/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── sales_intelligence.db   # Auto-created on first run
└── README.md
```

---

## 🔒 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Optional* | Groq API key (*can also be entered in the UI) |

---

## 🛠️ Troubleshooting

**ChromaDB error on first run**  
ChromaDB downloads the embedding model (`all-MiniLM-L6-v2`) on first use. Allow a minute for the download.

**Whisper not transcribing audio**  
Make sure `ffmpeg` is installed and accessible in your PATH.

**Short transcripts giving poor results**  
Very short transcripts (< 200 words) may produce only 1–2 chunks, reducing RAG quality. The sample transcript is a good length reference.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — fast LLM inference
- [LangChain](https://langchain.com) — RAG pipeline components
- [ChromaDB](https://trychroma.com) — vector store
- [OpenAI Whisper](https://github.com/openai/whisper) — audio transcription
- [Streamlit](https://streamlit.io) — UI framework
