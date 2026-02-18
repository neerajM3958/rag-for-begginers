from langchain_core.documents.base import Document
from langchain_text_splitters import CharacterTextSplitter
from app.util import get_logger

logger = get_logger()

def split_documents_using_character_text_splitter(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 0) -> list[Document]:
    logger.info("Split documents into smaller chunks with overlap")
    logger.info("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            logger.info(f"\n--- Chunk {i+1} ---")
            logger.info(f"Source: {chunk.metadata['source']}")
            logger.info(f"Length: {len(chunk.page_content)} characters")
            logger.info(f"Content:")
            logger.info(chunk.page_content)
            logger.info("-" * 50)
        
        if len(chunks) > 5:
            logger.info(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks