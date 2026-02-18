# RAG for Beginners

This project demonstrates Retrieval-Augmented Generation (RAG) using both OpenAI and Ollama models with LangChain and ChromaDB. It loads documents from the `docs/` folder, splits them into chunks, creates embeddings, and enables chat-based question answering over your data.

## Features
- Document loading from `docs/` (add your `.txt` files)
- Text chunking and embedding
- Vector store with ChromaDB
- Chat interface for Q&A
- Supports both OpenAI and Ollama models

## Requirements
- Python 3.12+
- (For Ollama) [Ollama](https://ollama.com/) installed and running locally
- (For OpenAI) OpenAI API key

## Installation
1. Clone the repository:
	```bash
	git clone <your-repo-url>
	cd rag-for-begginers
	```
2. (Recommended) Create a virtual environment:
	```bash
	python -m venv .venv
	source .venv/bin/activate  # On Windows: .venv\Scripts\activate
	```
3. Install dependencies:
	```bash
	pip install -e .
	```
4. Create a `.env` file in the project root (for OpenAI):
	```env
	OPENAI_API_KEY=your-openai-api-key
	```

## Usage

### 1. Prepare Documents
Place your `.txt` files in the `docs/` directory. These will be used as the knowledge base.

### 2. Run with Ollama (local LLM)
Make sure Ollama is installed and running. Then run:
```bash
python -m app.ollama.rag
```

### 3. Run with OpenAI (cloud LLM)
Make sure your `.env` file contains your OpenAI API key. Then run:
```bash
python -m app.open_ai.rag
```

## Logs
Logs are saved in the `logs/` directory with timestamps for each session.

## Project Structure
- `src/app/ollama/` — Ollama-based RAG pipeline
- `src/app/open_ai/` — OpenAI-based RAG pipeline
- `src/app/loaders/` — Document loaders
- `src/app/chunking/` — Text splitters
- `src/app/utils/` — Utilities (logging, chat)
- `db/` — ChromaDB vector stores
- `docs/` — Your source documents

## Notes
- The first run will create embeddings and a vector store. Subsequent runs will reuse the vector store unless documents change.
- To add new documents, place them in `docs/` and delete the corresponding `db/` subfolder to force re-embedding.

## License
MIT