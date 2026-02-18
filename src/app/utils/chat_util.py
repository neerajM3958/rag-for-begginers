from typing import List
from langchain_core.documents.base import Document
from app.utils.logging_util import get_logger
from langchain_core.messages import HumanMessage, SystemMessage

def simple_user_prompt(docs: List[Document], query: str):
    return f"""Based on the following documents, please answer this question: {query}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
def simple_system_prompt():
    return "You are a helpful assistant."
    
def start_chat(welcome_message: str, retriever, model, system_prompt: str = None):
    logger = get_logger()
    print(welcome_message)
    while True:
        query = input("> ")
        if not query.strip():
            continue
        if query.lower() in ["exit", "quit"]:
            logger.info("Exiting chat...")
            print("Goodbye!")
            break
        logger.info(f"User query: {query}")
        doc_chunks = retriever.invoke(query)
        # display results
        logger.info("--- Context ---")
        for i, doc in enumerate(doc_chunks, 1):
            logger.info(f"Document {i}:\n{doc.page_content}\n")

        combined_input = simple_user_prompt(doc_chunks, query)
        # Define the messages for the model
        messages = [
            SystemMessage(content=system_prompt or simple_system_prompt()),
            HumanMessage(content=combined_input),
        ]
        # Invoke the model with the combined input
        result = model.invoke(messages)

        logger.info("\n--- Generated Response ---")
        # logger.info("Full result:")
        # logger.info(result)
        logger.info(f"Content only: {result.content}")
        print(result.content)