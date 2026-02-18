import os
from typing import List
from langchain_core.documents.base import Document
import logging
from datetime import datetime

start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
default_logger_name = "app"

def simple_user_prompt(docs: List[Document], query: str):
    # Combine the query and the relevant document contents
    return f"""Based on the following documents, please answer this question: {query}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

def create_logger(tag: str, console_level: int = logging.INFO, logger_name: str = default_logger_name) -> logging.Logger:
    # create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # -------- File handler --------
    file_handler = logging.FileHandler(f"logs/{tag}_{start_timestamp}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_format)

    # -------- Console handler --------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    console_format = logging.Formatter(
        "%(levelname)s | %(message)s"
    )
    console_handler.setFormatter(console_format)

    # -------- Attach handlers --------
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def get_logger():
    return logging.getLogger(default_logger_name)
