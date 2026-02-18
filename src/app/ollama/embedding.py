import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from app.utils import get_logger

# mxbai-embed-large:335m
# nomic-embed-text

logger = get_logger()
model = "mxbai-embed-large:335m"
default_persist_directory = f"db/ollama_chroma_db"

def get_vector_store(persist_directory: str = default_persist_directory) -> Chroma:
    if os.path.exists(persist_directory):
        logger.info(f"Loading vector store from {persist_directory}")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OllamaEmbeddings(model=model),
            collection_metadata={"hnsw:space": "cosine"}
        )
    return None


def create_vector_store(chunks: list[Document], persist_directory: str = default_persist_directory) -> Chroma:
    logger.info("Creating embeddings and storing in ChromaDB...")
    
    # Check if vector store already exists
    vectorstore = get_vector_store(persist_directory)
    if vectorstore:
        logger.info(f"Vector store already exists at {persist_directory}")
        return vectorstore
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # Create ChromaDB vector store
    logger.info("--- Creating vector store ---")
    embedding_model = OllamaEmbeddings(model=model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    logger.info("--- Finished creating vector store ---")
    
    logger.info(f"Vector store created and saved to {persist_directory}")
    return vectorstore