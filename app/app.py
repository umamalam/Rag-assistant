"""
app.py -- chat interface.   Run with:  streamlit run app/app.py
"""

import os
import time

import streamlit as st

import ingest
import llm
import loaders
import rag_engine as engine
import store

UPLOAD_DIR = "/tmp/documents"

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="centered")
st.title("📚 Personal Knowledge Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------- sidebar: privacy status ----------
with st.sidebar:
    st.header("Privacy")
    if llm.is_local():
        st.success("Local mode: nothing leaves this machine.")
    else:
        msg = f"Cloud mode: retrieved excerpts are sent to {llm.provider_label()}."
        msg += (" Emails, phone numbers and card/SSN-like numbers are redacted first."
                if engine.redaction_enabled() else " Redaction is OFF.")
        st.warning(msg + " Set LLM_PROVIDER=ollama for fully local use.")

# ---------- sidebar: upload + index documents ----------
with st.sidebar:
    st.header("Add documents")
    exts = sorted(e.lstrip(".") for e in loaders.SUPPORTED)
    uploads = st.file_uploader(
        "Upload files", type=exts, accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads and st.button("Index files"):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        try:
            collection = store.get_collection()
            with st.spinner("Indexing..."):
                for f in uploads:
                    path = os.path.join(UPLOAD_DIR, os.path.basename(f.name))
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    ingest.ingest_file(collection, UPLOAD_DIR, path)
            st.session_state.uploader_key += 1  # clears the uploader
            st.rerun()
        except Exception as e:
            st.error(f"Indexing failed: {e}")

# ---------- load index ----------
try:
    available = engine.list_sources()
except Exception as e:
    st.error(f"Could not open the vector database: {e}")
    st.stop()

if not available:
    st.info("Nothing indexed yet. Upload a file from the sidebar and click **Index files**.")
    st.stop()

# ---------- sidebar: scope, summarize ----------
with st.sidebar:
    st.header("Scope")
    chosen = st.multiselect("Only search these files", list(available))
    window = st.selectbox("Document date", ["Any time", "Last 30 days", "Last 90 days", "Last year"])
    days = {"Last 30 days": 30, "Last 90 days": 90, "Last year": 365}.get(window)
    since_ts = time.time() - days * 86400 if days else None

    st.header("Summarize a document")
    target = st.selectbox("Document", list(available), key="sum_target")
    do_summary = st.button("Key takeaways")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.caption(f"📖 {sum(available.values())} chunks from {len(available)} files indexed")


def show_sources(sources):
    if sources:
        with st.expander("📎 Sources"):
            for s in sources:
                rel = f" (relevance {s['relevance']})" if s.get("relevance") is not None else ""
                st.markdown(f"**[{s['n']}]** {s['source']}, {s['location']}{rel}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))


def respond(user_text, fn):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                result = fn()
            except Exception as e:
                result = {"answer": f"Error: {e}", "sources": []}
        st.markdown(result["answer"])
        show_sources(result["sources"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]})


if do_summary:
    respond(f"Key takeaways from {target}", lambda: engine.summarize(target))

if question := st.chat_input("Ask something about your documents..."):
    history = list(st.session_state.messages)  # before this turn is appended
    respond(question, lambda: engine.ask(
        question, history=history, sources=chosen or None, since_ts=since_ts))