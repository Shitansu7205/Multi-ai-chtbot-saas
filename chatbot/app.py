import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Your existing RAG imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


# ==========================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# STEP 2: CREATE FASTAPI APP
# ==========================================

app = FastAPI()


@lru_cache(maxsize=1)
def get_rag_pipeline():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "CAREER_counsellor.pdf"
    if not pdf_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Required PDF not found: {pdf_path}",
        )
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured.",
        )

    raw_documents = PyPDFLoader(str(pdf_path)).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(raw_documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_database = Chroma.from_documents(split_docs, embeddings)
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

    system_instruction = (
        "You are a helpful assistant. Use the provided context below to answer "
        "the user's question. If you don't know the answer based on the context, "
        "honestly say that you don't know. Do not make things up.\n\n"
        "Context:\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{input}"),
    ])
    retriever = vector_database.as_retriever(search_kwargs={"k": 3})
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, document_chain)


# ==========================================
# STEP 11: REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):
    question: str


# ==========================================
# STEP 12: TEST ROUTE
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Career Counsellor RAG API is running"
    }


# ==========================================
# STEP 13: CHAT ROUTE
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):
    rag_pipeline = get_rag_pipeline()
    result = rag_pipeline.invoke({
        "input": request.question
    })

    answer = result["answer"]

    return {
        "question": request.question,
        "answer": answer
    }








