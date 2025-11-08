# AskADS - MS in Applied Data Science RAG Bot

A Retrieval Augmented Generation (RAG) chatbot application that provides intelligent answers about the University of Chicago MS in Applied Data Science program. Built with Streamlit, ChromaDB, and OpenAI.

## Features

- **Hybrid Retrieval**: Combines dense (E5 embeddings) and sparse (TF-IDF) retrieval methods
- **Intelligent Reranking**: Optional cross-encoder reranking for improved relevance
- **Context-Aware Answers**: Generates responses using GPT-4o-mini with retrieved context
- **Source Citations**: Provides numbered citations with links to original documents
- **PII Protection**: Automatically redacts emails and phone numbers from responses
- **Custom UI**: Responsive interface with University of Chicago maroon theme

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "MS ADS RAG BOT"
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or provide it in the sidebar when running the app.

5. **Ensure RAG index exists**:
   The application expects a `rag_index/` directory containing:
   - `chroma_db/` - ChromaDB vector database
   - `meta.jsonl` - Document metadata (optional)

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser. Enter questions about the MS-ADS program in the chat interface.

## Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: E5-base-v2 (sentence-transformers)
- **Reranking**: BAAI/bge-reranker-base (optional)
- **LLM**: OpenAI GPT-4o-mini
- **Retrieval**: Hybrid dense + sparse (TF-IDF) with MMR diversity

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── rag_index/            # RAG index directory
│   ├── chroma_db/        # ChromaDB vector store
│   └── meta.jsonl        # Document metadata
└── README.md             # This file
```

## Requirements

- Python 3.8+
- OpenAI API key
- RAG index files in `rag_index/` directory

---

**Note**: This project was developed for the University of Chicago MS in Applied Data Science program as part of a Generative AI course.

