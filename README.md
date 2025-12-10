# Multimodal RAG System (Document-Level)
A Multimodal Retrieval-Augmented Generation (RAG) system that processes PDFs containing text, tables, and images, extracts structured content, generates summaries, stores embeddings into ChromaDB, and answers user queries in real time using OpenAI GPT-4o.

---

## 🚀 Features

### ✅ Multimodal PDF Processing
- Extracts **text, tables, and images** using `unstructured`.
- Converts pages to images via **PyMuPDF**.
- Captures **source page numbers and image paths** for transparent citations.

### ✅ Intelligent Chunking
- Uses **title-based chunking** (`chunk_by_title`) for coherent sections.
- Supports **mixed-content chunks** (text + tables + images).

### ✅ AI-Generated Summaries
- GPT-4o creates **searchable, grounded summaries** for each chunk.
- Strict prompts ensure: **no hallucinations**, only document-grounded content.

### ✅ Vector Storage with ChromaDB
- Each uploaded PDF is stored in its **own Chroma collection**.
- Enables scalable **multi-document retrieval**.

### ✅ Real-Time Question Answering
- Retrieves the most relevant chunks using **embedding similarity**.
- GPT-4o produces concise answers based solely on retrieved content.
- Enforces: **“If it’s not in the documents → I don’t know.”**

### ✅ FastAPI Backend
- `POST /process_pdf/` → Upload + process + store PDF as a Chroma collection  
- `POST /get_answer/` → Query a specific document’s vector store for answers

---

## 🧱 Tech Stack

- **Python**, **FastAPI**
- **Unstructured** (PDF parsing)
- **PyMuPDF** (page-to-image conversion)
- **LangChain**
- **OpenAI (GPT-4o & embeddings)**
- **ChromaDB** (vector storage)

---

# 📦 Installation

## 1. Clone the repository
```bash
git clone <repo-url>
cd <repo>
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Set environment variables
Create a `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

---

# ▶️ Running the FastAPI Server
```bash
uvicorn app:app --reload
```

FastAPI will be available at:

👉 http://127.0.0.1:8000

---

# API Endpoints

### **1. `/process_pdf/` — Upload & Index PDF**
**Method:** `POST`

This endpoint processes a PDF, extracts text/tables/images, chunks it, creates embeddings, and stores everything inside a dedicated ChromaDB collection.

#### **Request Body**
```json
{
  "pdf_name": "Attention-is-all-you-need.pdf"
}
```

#### **Response**

```json
{
  "status": "success",
  "collection_name": "Attention-is-all-you-need",
  "message": "PDF processed and vector store created."
}
```

### **2. `/get_answer/` — Ask Questions**
**Method:** `POST`

Retrieves relevant chunks from the document’s ChromaDB collection and generates an answer grounded strictly in the retrieved content.

#### **Request Body**
```json
{
  "pdf_name": "Attention-is-all-you-need.pdf",
  "question": "what is a transformer?",
  "collection_name": "Attention-is-all-you-need"
}
```

#### **Response**

```json
{
    "status": "success",
    "answer": {
        "final_answer": "The Transformer model is a sequence transduction architecture that relies entirely on attention mechanisms, specifically multi-headed self-attention, instead of recurrent layers. It consists of an encoder-decoder structure where the encoder maps input sequences to continuous representations, and the decoder generates output sequences auto-regressively. The model uses multi-head attention in three ways: encoder-decoder attention, self-attention in the encoder, and masked self-attention in the decoder to maintain the auto-regressive property. This architecture allows for significant parallelization, improving training speed and performance, particularly in translation tasks. The Transformer has achieved state-of-the-art results in translation and is being extended to other modalities and tasks.",
        "page_links": [
            "./pdf_images/Attention-is-all-you-need/page_2.png",
            "./pdf_images/Attention-is-all-you-need/page_3.png",
            "./pdf_images/Attention-is-all-you-need/page_5.png",
            "./pdf_images/Attention-is-all-you-need/page_10.png"
        ]
    }
}
```

# How Retrieval Works

- Each PDF becomes its own ChromaDB collection.
- Each chunk includes rich metadata:
  **original text**
  **extracted tables**
  **images(base64)**
  **page numbers**

- OpenAI embeddings are used for semantic search.
- GPT-4o generates grounded answers only from retrieved chunks.
- Returned answers always include the source page images.





