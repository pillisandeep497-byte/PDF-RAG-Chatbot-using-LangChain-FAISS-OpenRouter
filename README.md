# 📚 PDF RAG Chatbot using LangChain, FAISS & OpenRouter

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about PDF documents.

The application:

* Loads PDF documents
* Splits text into chunks
* Converts chunks into vector embeddings
* Stores embeddings in a FAISS vector database
* Retrieves the most relevant chunks
* Uses an LLM via OpenRouter to generate answers based only on retrieved context

This helps reduce hallucinations and ensures answers come directly from the provided documents.

---

## 🛠️ Tech Stack

* Python
* LangChain
* FAISS
* HuggingFace Embeddings
* OpenRouter
* GPT-OSS-20B
* PyPDF
* dotenv

---

## 📂 Project Structure

```text
pdf-rag-chatbot-langchain-faiss/
│
├── RAG/
│   └── sample2.pdf
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## ⚡ Features

✅ PDF document loading

✅ Intelligent text chunking

✅ Semantic search using vector embeddings

✅ FAISS vector database

✅ Context-aware question answering

✅ Hallucination reduction

✅ OpenRouter LLM integration

---

## 🔄 RAG Pipeline

```text
PDF
 │
 ▼
Document Loader
 │
 ▼
Text Splitter
 │
 ▼
Embeddings Model
 │
 ▼
FAISS Vector Store
 │
 ▼
Retriever
 │
 ▼
Retrieved Context
 │
 ▼
LLM (GPT-OSS-20B)
 │
 ▼
Final Answer
```

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot-langchain-faiss.git

cd pdf-rag-chatbot-langchain-faiss
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

## ▶️ Run Project

```bash
python main.py
```

Example:

```text
enter: What is Artificial Intelligence?

answer:
Artificial Intelligence is...
```

---

## 📖 How It Works

### Step 1: Load PDF

```python
loader = PyPDFLoader("RAG/sample2.pdf")
documents = loader.load()
```

### Step 2: Split Text

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
```

### Step 3: Create Embeddings

```python
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```

### Step 4: Store in FAISS

```python
vector_store = FAISS.from_documents(...)
```

### Step 5: Retrieve Relevant Chunks

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k":15}
)
```

### Step 6: Generate Response

The LLM receives only the retrieved document context and answers accordingly.

---

## 🎯 Learning Outcomes

By building this project, you learn:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embeddings
* Semantic Search
* Prompt Engineering
* LangChain Chains
* FAISS Integration
* OpenRouter APIs
* Context Injection
* LLM Application Development

---

## 🔮 Future Improvements

* Streamlit UI
* Chat History
* Multiple PDF Support
* Hybrid Search
* Metadata Filtering
* Source Citation
* Persistent Vector Database
* Conversational RAG

---

## 👨‍💻 Author

Sandeep Pilli

AI & ML Diploma Student

Passionate about Generative AI, LangChain, RAG Systems, and LLM Applications.
