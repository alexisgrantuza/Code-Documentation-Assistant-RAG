import os
import pickle
from typing import List, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from config.settings import settings

class VectorDB:
  def __init__(self):
    self.embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key
    )
    self.vectorstore = None
    self.db_path = settings.vector_db_path
  
  def create_vectorstore(self, documents: List[Document]) -> bool:
    """Create vector store from documents"""
    try:
        self.vectorstore = FAISS.from_documents(
            documents, 
            self.embeddings
        )
        return True
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False
  
  def save_vectorstore(self, repo_name: str):
    """Save vector store to disk"""
    if self.vectorstore:
        os.makedirs(self.db_path, exist_ok=True)
        save_path = os.path.join(self.db_path, f"{repo_name}.faiss")
        self.vectorstore.save_local(save_path)
  
  def load_vectorstore(self, repo_name: str) -> bool:
    """Load vector store from disk"""
    try:
        load_path = os.path.join(self.db_path, f"{repo_name}.faiss")
        if os.path.exists(load_path):
            self.vectorstore = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        return False
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return False
  
  def similarity_search(self, query: str, k: int = 4) -> List[Document]:
    """Search for similar documents"""
    if self.vectorstore:
        return self.vectorstore.similarity_search(query, k=k)
    return []

  def add_documents(self, documents: List[Document]):
    """Add new documents to existing vector store"""
    if self.vectorstore:
        self.vectorstore.add_documents(documents)