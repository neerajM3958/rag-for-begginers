from .text_loader import load_text_documents

def load_documents(docs_path="docs"):
    return load_text_documents(docs_path)

__all__ = ["load_documents","load_text_documents"]
