# app.py
import os
import io
import hashlib
import tempfile
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

# ---------- ENV ----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Book Bot", page_icon="📚")
st.title("📚 Book Bot – AI Study Buddy")

if not API_KEY:
    st.error("GOOGLE_API_KEY missing in .env")
    st.stop()

# ---------- MODELS ----------
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-1.5-flash"

embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=API_KEY,
    temperature=0.2,
    streaming=True,  # stream tokens for snappy UX
)

# ---------- HELPERS ----------
def _file_bytes(f) -> bytes:
    # Read the uploaded file fully without mutating its internal pointer permanently
    pos = f.tell()
    f.seek(0)
    data = f.read()
    f.seek(pos)
    return data

def files_fingerprint(files) -> str:
    """Stable fingerprint for the *content* of uploaded files (order-insensitive)."""
    hasher = hashlib.sha256()
    blobs = []
    for f in files:
        name = os.path.basename(f.name)
        ext = os.path.splitext(name)[1].lower()
        blobs.append((ext, _file_bytes(f)))
    # sort by ext+sha to make order irrelevant
    parts = []
    for ext, blob in blobs:
        parts.append(ext + ":" + hashlib.sha256(blob).hexdigest())
    parts.sort()
    hasher.update("|".join(parts).encode("utf-8"))
    return hasher.hexdigest()

def load_docs(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        name = os.path.basename(f.name)
        ext = os.path.splitext(name)[1].lower()
        # dump to a temp file so community loaders can read
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            data = _file_bytes(f)
            tmp.write(data)
            path = tmp.name
        if ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        elif ext in [".txt", ".md"]:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        else:
            st.warning(f"Unsupported file type: {ext}. Skipping {name}")
    return docs

@st.cache_resource(show_spinner=False)
def _split_docs(_docs: List[Document]):
    """Cache only the split step; underscore prevents hashing of complex objs."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(_docs)

def build_or_load_faiss(_chunks: List[Document], fp: str):
    """
    Load FAISS index from ./indices/{fp} if present, else build and persist.
    Kept OUT of cache on purpose; the on-disk index is the cache.
    """
    idx_dir = os.path.join("indices", fp)
    os.makedirs("indices", exist_ok=True)
    if os.path.isdir(idx_dir) and any(
        fn.endswith(".faiss") or fn.endswith(".pkl") for fn in os.listdir(idx_dir)
    ):
        vs = FAISS.load_local(
            idx_dir, embeddings, allow_dangerous_deserialization=True
        )
        return vs, False  # loaded
    # Build with a visible progress bar
    with st.spinner("Indexing documents (one-time)…"):
        vs = FAISS.from_documents(_chunks, embeddings)
        vs.save_local(idx_dir)
    return vs, True  # built

# ---------- PROMPT / CHAIN ----------
PROMPT = ChatPromptTemplate.from_template(
    """You are Book Bot, a helpful study assistant.
Answer ONLY using the provided context. Cite sources as [1], [2], etc.
If you don’t know, say so.

# Context
{context}

# Question
{question}
"""
)

def render_context(docs: List[Document]) -> str:
    return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])

def make_chain(retriever):
    return (
        RunnableMap(
            {
                "context": lambda x: render_context(
                    retriever.get_relevant_documents(x["question"])
                ),
                "question": lambda x: x["question"],
            }
        )
        | PROMPT
        | llm
    )

# ---------- (Optional) SPEECH ----------
def speak(text: str):
    """Best-effort TTS: uses gTTS if installed; otherwise no-op with hint."""
    try:
        from gtts import gTTS  # local import so missing pkg doesn't break app
        tts = gTTS(text)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.info("Install `gTTS` to enable speech: `pip install gTTS`")
        st.caption(f"TTS skipped ({type(e).__name__}).")

# ---------- UI ----------
files = st.file_uploader(
    "Upload PDF/TXT/MD (multiple allowed)", type=["pdf", "txt", "md"], accept_multiple_files=True
)

if files:
    # Fingerprint by content to re-use on-disk FAISS index across sessions
    fp = files_fingerprint(files)
    docs = load_docs(files)
    chunks = _split_docs(docs)

    # Build or load FAISS (persisted)
    vs, built = build_or_load_faiss(chunks, fp)
    if built:
        st.success("Index built and cached ✅")
    else:
        st.caption("Loaded cached index ⚡")

    retriever = vs.as_retriever(search_kwargs={"k": 4})
    chain = make_chain(retriever)

    q = st.text_input("Ask something about your book")
    if q:
        st.subheader("Answer")
        container = st.empty()
        final_answer = []

        # Stream tokens as they arrive
        for chunk in chain.stream({"question": q}):
            if hasattr(chunk, "content") and chunk.content:
                final_answer.append(chunk.content)
                container.markdown("".join(final_answer))

        # Speak (optional)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔊 Speak"):
                speak("".join(final_answer))
        with col2:
            if st.button("↻ Clear"):
                st.rerun()
else:
    st.info("Upload at least one PDF/TXT/MD to get started.")