import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents.base import Document
from app.util import get_logger

logger = get_logger()

def load_text_documents(docs_path="docs"):
    logger.info("Load all text files from the docs directory")
    logger.info(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory (UTF-8 to avoid Windows cp1252 decode errors)
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        logger.info(f"\nDocument {i+1}:")
        logger.info(f"  Source: {doc.metadata['source']}")
        logger.info(f"  Content length: {len(doc.page_content)} characters")
        logger.info(f"  Content preview: {doc.page_content[:100]}...")
        logger.info(f"  metadata: {doc.metadata}")
    return documents