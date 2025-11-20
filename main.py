# ============================
# 📌 RAG PIPELINE WITH LANGGRAPH
# ============================

# -------- IMPORTS --------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

# ============================
# 📌 1. LOAD PDF
# ============================

def load_local_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


docs = load_local_pdf("pdf_files/Sample-pdf.pdf")
print("Total pages loaded:", len(docs))


# ============================
# 📌 2. CHUNKING
# ============================

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ","]
    )
    return splitter.split_documents(docs)


chunks = chunk_docs(docs)
print("Total chunks:", len(chunks))
print("Example chunk:", chunks[0].page_content[:300])


# ============================
# 📌 3. EMBEDDINGS (FAST MODEL)
# ============================

def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = create_embeddings()
print("Embedding dimension:", len(embeddings.embed_query("hello")))


# ============================
# 📌 4. CHROMA VECTOR DB
# ============================

def create_vector_db(chunks, embeddings):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_store"
    )
    vectordb.persist()
    return vectordb


vectordb = create_vector_db(chunks, embeddings)
print("Vector DB created.")


# ============================
# 📌 5. BM25 SPARSE RETRIEVER
# ============================

def build_bm25_index(chunks):
    corpus = [doc.page_content for doc in chunks]
    tokenized = [text.split(" ") for text in corpus]
    return BM25Okapi(tokenized)


bm25 = build_bm25_index(chunks)


# ============================
# 📌 6. HYBRID SEARCH TEST
# ============================

def hybrid_search(query, vectordb, bm25, top_k=5):
    dense_results = vectordb.similarity_search(query, k=top_k)

    tokenized_query = query.split(" ")
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_idx = sorted(
        list(enumerate(sparse_scores)),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    sparse_docs = [chunks[i] for i, _ in sparse_idx]
    return dense_results + sparse_docs


results = hybrid_search("billing policy rules", vectordb, bm25, top_k=5)
print("Hybrid search returned:", len(results))


# ============================
# 📌 7. LANGGRAPH WORKFLOW
# ============================

class RAGState(TypedDict):
    query: str
    vectordb: any
    bm25: any
    chunks: List[Document]
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    validated_docs: List[Document]
    answer: str


# ----------- NODES -----------

def retrieval_node(state: RAGState):
    query = state["query"]
    vectordb = state["vectordb"]
    bm25 = state["bm25"]
    chunks = state["chunks"]

    # Dense search
    dense_results = vectordb.similarity_search(query, k=5)

    # Sparse search
    tokenized_query = query.split(" ")
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_idx = sorted(
        list(enumerate(sparse_scores)),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    sparse_results = [chunks[i] for i, _ in sparse_idx]

    hybrid = dense_results + sparse_results

    # Remove duplicates
    seen = set()
    unique_docs = []
    for doc in hybrid:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    state["retrieved_docs"] = unique_docs[:5]
    return state


# Reranker model
reranker = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def rerank_node(state: RAGState):
    query = state["query"]
    docs = state["retrieved_docs"]

    query_emb = reranker.encode(query)
    doc_embs = reranker.encode([d.page_content for d in docs])

    scores = util.cos_sim(query_emb, doc_embs)[0]

    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    state["reranked_docs"] = [d for _, d in ranked_docs]
    return state


def validation_node(state: RAGState):
    filtered = [d for d in state["reranked_docs"] if len(d.page_content.strip()) > 0]
    state["validated_docs"] = filtered
    return state


# Groq client
groq_api_key=os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)


def answer_generation_node(state: RAGState):
    context = "\n\n".join(d.page_content for d in state["validated_docs"])

    prompt = f"""
You are a helpful assistant.
Use ONLY the following context to answer the question.

CONTEXT:
{context}

QUESTION:
{state["query"]}

Provide a short and clear final answer.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    state["answer"] = response.choices[0].message.content
    return state


# ============================
# 📌 8. BUILD WORKFLOW GRAPH
# ============================

def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("reranking", rerank_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("answer_generation", answer_generation_node)

    workflow.add_edge("retrieval", "reranking")
    workflow.add_edge("reranking", "validation")
    workflow.add_edge("validation", "answer_generation")
    workflow.add_edge("answer_generation", END)

    workflow.set_entry_point("retrieval")

    return workflow.compile()


graph = build_rag_graph()


# ============================
# 📌 9. RUN QUERY
# ============================

response = graph.invoke({
    "query": "What is the refund policy?",
    "vectordb": vectordb,
    "bm25": bm25,
    "chunks": chunks
})

print("\n====================")
print("FINAL ANSWER")
print("====================")
print(response["answer"])
