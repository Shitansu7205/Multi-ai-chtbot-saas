from pathlib import Path

from fastapi import FastAPI
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


# ==========================================
# STEP 3: LOAD PDF
# ==========================================

pdf_path = Path(__file__).resolve().parent.parent / "data" / "CAREER_counsellor.pdf"
loader = PyPDFLoader(str(pdf_path))

raw_documents = loader.load()

print(f"Successfully loaded {len(raw_documents)} pages from PDF.")


# ==========================================
# STEP 4: SPLIT DOCUMENT
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(raw_documents)

print(f"Created {len(split_docs)} chunks.")


# ==========================================
# STEP 5: CREATE EMBEDDINGS + CHROMA
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_database = Chroma.from_documents(
    split_docs,
    embeddings
)


# ==========================================
# STEP 6: CREATE GROQ LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)


# ==========================================
# STEP 7: CREATE PROMPT
# ==========================================

system_instruction = (
    "You are a helpful assistant. Use the provided context below to answer "
    "the user's question. If you don't know the answer based on the context, "
    "honestly say that you don't know. Do not make things up.\n\n"
    "Context:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "{input}")
])


# ==========================================
# STEP 8: CREATE RETRIEVER
# ==========================================

retriever = vector_database.as_retriever(
    search_kwargs={"k": 3}
)


# ==========================================
# STEP 9: CREATE DOCUMENT CHAIN
# ==========================================

document_chain = create_stuff_documents_chain(
    llm,
    prompt_template
)


# ==========================================
# STEP 10: CREATE RAG PIPELINE
# ==========================================

rag_pipeline = create_retrieval_chain(
    retriever,
    document_chain
)


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

    result = rag_pipeline.invoke({
        "input": request.question
    })

    answer = result["answer"]

    return {
        "question": request.question,
        "answer": answer
    }









