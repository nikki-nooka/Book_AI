import os
import tempfile
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document

from langgraph.graph import StateGraph, END

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
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=API_KEY, temperature=0.2)

# ---------- HELPERS ----------
def load_docs(files) -> List[Document]:
    docs = []
    for f in files:
        ext = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.read())
            path = tmp.name
        if ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        elif ext in [".txt", ".md"]:
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs

def build_vs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# ---------- LANGGRAPH ----------
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

PROMPT = """You are Book Bot, a helpful study assistant.
Answer ONLY using the provided context. Cite as [1], [2], etc.
If you don’t know, say so."""

def render(docs):
    return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])

def retrieve_node(retriever):
    def _fn(state: State):
        return {"context": retriever.get_relevant_documents(state["question"])}
    return _fn

def generate_node():
    def _fn(state: State):
        ctx = render(state["context"])
        prompt = f"{PROMPT}\n\n# Context\n{ctx}\n\n# Question\n{state['question']}"
        ans = llm.invoke(prompt).content
        return {"answer": ans}
    return _fn

def build_graph(retriever):
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node(retriever))
    g.add_node("generate", generate_node())
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()

# ---------- UI ----------
files = st.file_uploader("Upload PDF/TXT/MD", type=["pdf","txt","md"], accept_multiple_files=True)

if files:
    docs = load_docs(files)
    vs = build_vs(docs)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    graph = build_graph(retriever)

    q = st.text_input("Ask something about your book")
    if st.button("Ask") and q:
        result = graph.invoke({"question": q})
        st.subheader("Answer")
        st.write(result["answer"])
