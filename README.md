# Personal Knowledge Assistant (RAG)

A retrieval-augmented question-answering system over your own documents.
Ask questions in natural language, get answers grounded in what your
documents actually say, with page-level source citations.

## Architecture

```
documents/  (PDF, MD, TXT, DOCX, EML, MBOX -- subfolders OK)
      |  loaders.py: split into natural units (page / section / email)
      v
ingest.py: 800-char chunks -> LOCAL embeddings (all-MiniLM-L6-v2)
      |    content-hash tracking: skip unchanged, re-index edited files
      v
ChromaDB on disk (cosine distance, telemetry off)
      |
[question] -> rewrite follow-ups into standalone query (chat memory)
      -> retrieve 30 candidates -> relevance floor -> diversify across sources
      -> (optional PII redaction) -> LLM: Groq (cloud) or Ollama (fully local)
      -> answer with [n] citations -> Streamlit chat UI
```

## Setup

```bash
cd app
pip install -r requirements.txt
export GROQ_API_KEY="your-key"      # or use local mode, below
```

## Usage

```bash
# put files in ../documents/, then:
python3 ingest.py                    # index new/changed files
streamlit run app.py                 # chat UI (bound to localhost only)
python3 rag_engine.py "your question"
python3 test_pipeline.py             # offline tests, no model or API needed
python3 eval.py                      # your own known-answer test cases
```

In the UI: chat with follow-ups, restrict a question to specific files or a
date window (sidebar), or click **Key takeaways** to summarize a whole document.

## What makes it more than search

- **Cross-source synthesis**: retrieval caps chunks per source (default 3 of 6),
  so answers combine several documents; the prompt asks the model to note where
  sources agree or conflict.
- **Whole-document summaries**: `summarize()` samples chunks evenly across a
  document. Top-K search alone can never answer "key takeaways from that book".
- **Conversation memory**: "what about the second point?" is rewritten into a
  standalone query before retrieval.
- **Honest "I don't know"**: if nothing clears the relevance floor
  (`MIN_RELEVANCE`, default 0.15), the LLM is not called at all.

## Privacy and security

| Concern | What's done |
|---|---|
| Data stays local | Embeddings and vector DB run on your machine. |
| LLM calls | `LLM_PROVIDER=ollama` keeps everything local. With Groq, retrieved excerpts are sent to Groq. The UI says which mode is active. |
| PII to the cloud | With Groq, emails, phone numbers, card/SSN-like numbers and API keys are redacted from excerpts and the question first (`REDACT_PII=0` to disable). Regex-based, best effort, not a guarantee. |
| Prompt injection | Emails and PDFs are untrusted. The system prompt tells the model never to follow instructions found inside excerpts. |
| Deletion | `python3 ingest.py --remove NAME` or `--wipe`. |
| Exposure | Streamlit binds to localhost; usage stats and Chroma telemetry are off; `documents/`, `data/`, `.env` are git-ignored. |
| Not done | Encryption at rest. Use FileVault/an encrypted volume for `data/`. |

Local mode: `ollama pull llama3.1:8b`, then `export LLM_PROVIDER=ollama`.

## Known limits

- Scanned/image PDFs have no text layer (ingest warns; OCR not included).
- Email "date" filtering uses the sent date; for other files it uses file
  modified time, which is when you saved/downloaded it, not when you read it.
- `MIN_RELEVANCE`, `TOP_K`, `MAX_PER_SOURCE` are sensible defaults, not tuned to
  your data. Tune them with `eval.py` on questions you know the answers to.
- Summaries are based on sampled excerpts (`SUMMARY_CHUNKS`, default 16), sized
  for Groq's free-tier token limits. Raise it with Ollama.
- The offline tests use a stand-in embedder, so they verify the pipeline logic,
  not the retrieval quality of the real model on your documents.

## Possible extensions

Hybrid BM25 + semantic search, a reranker, OCR, map-reduce summaries for very
long books, Apple Notes / Notion / browser-history importers.
