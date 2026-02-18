from typing import List
from langchain_core.documents.base import Document
from app.utils.logging_util import get_logger
from langchain_core.messages import HumanMessage, SystemMessage

def simple_user_prompt(docs: List[Document], query: str):
    formatted_docs = "\n\n".join(
        [
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ]
    )

    return f"""
    Answer the question using the provided documents.

    Question:
    {query}

    Documents:
    {formatted_docs}

    Instructions:
    - Base your answer only on the documents.
    - If the answer is not contained in the documents, respond with the refusal sentence exactly as specified.
    """
    
def simple_system_prompt():
    return """
    You are a question-answering assistant that must use ONLY the provided documents.

    Rules:
    - Use only the information present in the documents.
    - Do NOT use prior knowledge.
    - Do NOT infer beyond what is written.
    - If the answer is partially available, answer only the supported part.
    - If the answer is not present, say:

    "I don't have enough information to answer that question based on the provided documents."

    - Be concise, factual, and avoid repetition.
    - Do not mention the documents explicitly in your answer.
    """
    
def start_chat(welcome_message: str, retriever, model, system_prompt: str = None):
    logger = get_logger()
    print(welcome_message)
    while True:
        query = input("> ")
        if not query.strip():
            continue
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
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