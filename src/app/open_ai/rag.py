from langchain_core.documents.base import Document
from app.util import simple_user_prompt
from app.loaders import load_documents
from app.chunking import split_documents_using_character_text_splitter
from .embedding import create_vector_store, get_vector_store
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import dotenv
from app.util import create_logger
import logging


dotenv.load_dotenv()

model = "gpt-4o"
welcome_message = """
Hey, I'm RAG chat assistant.
training data: local documents
library: langchain-openai
model: gpt-4o
To exit, type 'exit' or 'quit'.
"""

def main():
    logger = create_logger("open_ai_rag", console_level=logging.WARNING)
    
    vectorstore = get_vector_store()
    if not vectorstore:
        documents = load_documents()
        chunks = split_documents_using_character_text_splitter(documents)
        vectorstore = create_vector_store(chunks)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")
    
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
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
        ]
        # Invoke the model with the combined input
        result = model.invoke(messages)

        logger.info("\n--- Generated Response ---")
        # logger.info("Full result:")
        # logger.info(result)
        logger.info(f"Content only: {result.content}")
        print(result.content)
if __name__ == "__main__":
    main()

    