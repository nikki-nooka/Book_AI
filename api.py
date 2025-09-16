# api.py
import os
import hashlib
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

# ----------- ENV -----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# ----------- FastAPI init -----------
app = FastAPI(title="Book Bot API", version="1.0")

# ----------- MODELS -----------
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-1.5-flash"

embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=API_KEY, temperature=0.2)

# ----------- GLOBAL STATE -----------
INDEX_DIR = "./indexes"
os.makedirs(INDEX_DIR, exist_ok=True)
vectorstores = {}  # in-memory cache


# ----------- HELPERS -----------
def file_bytes(f: UploadFile) -> bytes:
    """Read file bytes and reset pointer so it can be reused."""
    data = f.file.read()
    f.file.seek(0)
    return data


def files_fingerprint(files: List[UploadFile]) -> str:
    hasher = hashlib.sha256()
    parts = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        data = file_bytes(f)
        parts.append(ext + ":" + hashlib.sha256(data).hexdigest())
    parts.sort()
    hasher.update("|".join(parts).encode("utf-8"))
    return hasher.hexdigest()


def load_docs(files: List[UploadFile]) -> List[Document]:
    docs = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        data = file_bytes(f)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp.flush()
            path = tmp.name

        try:
            if ext == ".pdf":
                docs.extend(PyPDFLoader(path).load())
            elif ext in [".txt", ".md"]:
                docs.extend(TextLoader(path, encoding="utf-8").load())
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        finally:
            os.unlink(path)  # cleanup temp file

    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def render_context(docs: List[Document]) -> str:
    return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])


def make_chain(retriever):
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
    return (
        RunnableMap(
            {
                "context": lambda x: render_context(
                    retriever.invoke(x["question"])  # fixed: invoke instead of deprecated get_relevant_documents
                ),
                "question": lambda x: x["question"],
            }
        )
        | PROMPT
        | llm
    )


def save_index(fp: str, vs: FAISS):
    vs.save_local(os.path.join(INDEX_DIR, fp))


def load_index(fp: str) -> FAISS:
    return FAISS.load_local(os.path.join(INDEX_DIR, fp), embeddings, allow_dangerous_deserialization=True)


# ----------- ENDPOINTS -----------

@app.get("/health")
def health():
    return {"status": "ok", "msg": "Book Bot API running"}


@app.post("/upload")
async def upload(files: List[UploadFile]):
    try:
        fp = files_fingerprint(files)

        # check cache
        if fp in vectorstores:
            return {"msg": "Index already in memory", "fingerprint": fp}

        # check disk
        index_path = os.path.join(INDEX_DIR, fp)
        if os.path.exists(index_path):
            vs = load_index(fp)
            vectorstores[fp] = vs
            return {"msg": "Index loaded from disk", "fingerprint": fp}

        # build new
        docs = load_docs(files)
        chunks = split_docs(docs)
        vs = FAISS.from_documents(chunks, embeddings)

        save_index(fp, vs)
        vectorstores[fp] = vs

        return {"msg": "Index built", "fingerprint": fp}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ask")
async def ask(fingerprint: str = Form(...), question: str = Form(...)):
    if fingerprint not in vectorstores:
        index_path = os.path.join(INDEX_DIR, fingerprint)
        if os.path.exists(index_path):
            vs = load_index(fingerprint)
            vectorstores[fingerprint] = vs
        else:
            return JSONResponse({"error": "Fingerprint not found. Upload first."}, status_code=400)

    vs = vectorstores[fingerprint]
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    chain = make_chain(retriever)

    final_answer = []
    for chunk in chain.stream({"question": question}):
        if hasattr(chunk, "content") and chunk.content:
            final_answer.append(chunk.content)

    return {"answer": "".join(final_answer), "fingerprint": fingerprint}


# ----------- SIMPLE HTML UI -----------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Book Bot</title></head>
        <body style="font-family:Arial; margin:40px;">
            <h2>Book Bot</h2>
            <form action="/chat" method="get">
                <label>Fingerprint:</label><br>
                <input type="text" name="fingerprint" style="width:400px"><br><br>
                <label>Question:</label><br>
                <input type="text" name="question" style="width:400px"><br><br>
                <button type="submit">Ask</button>
            </form>
        </body>
    </html>
    """


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(fingerprint: str = Query(...), question: str = Query(...)):
    if fingerprint not in vectorstores:
        index_path = os.path.join(INDEX_DIR, fingerprint)
        if os.path.exists(index_path):
            vs = load_index(fingerprint)
            vectorstores[fingerprint] = vs
        else:
            return HTMLResponse("<h3>Error: Fingerprint not found. Upload first.</h3>", status_code=400)

    vs = vectorstores[fingerprint]
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    chain = make_chain(retriever)

    final_answer = []
    for chunk in chain.stream({"question": question}):
        if hasattr(chunk, "content") and chunk.content:
            final_answer.append(chunk.content)

    answer = "".join(final_answer)

    return f"""
    <html>
        <head><title>Book Bot Result</title></head>
        <body style="font-family:Arial; margin:40px;">
            <h2>Book Bot</h2>
            <p><b>Question:</b> {question}</p>
            <p><b>Answer:</b> {answer}</p>
        </body>
    </html>
    """
