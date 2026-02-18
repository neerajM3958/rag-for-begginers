from app.loaders import load_documents
from app.chunking import split_text_using_recursive_character_splitter
from .embedding import create_vector_store, get_vector_store
from langchain_openai import ChatOpenAI
import dotenv
from app.utils import create_logger, start_chat
import logging


dotenv.load_dotenv()

model_name = "gpt-4o"
welcome_message = f"""
Hey there! I'm your RAG assistant. I can answer questions based on the provided documents. 
Just type your question and I'll do my best to help you out using the information from the documents.
training data: local documents
library: langchain-openai
model: {model_name}
To exit, type 'exit' or 'quit'.
"""

def main():
    create_logger("open_ai_rag", console_level=logging.WARNING)
    vectorstore = get_vector_store()
    if not vectorstore:
        documents = load_documents()
        chunks = split_text_using_recursive_character_splitter(documents)
        vectorstore = create_vector_store(chunks)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    model = ChatOpenAI(model=model_name)
    start_chat(welcome_message, retriever, model)
    
if __name__ == "__main__":
    main()

    