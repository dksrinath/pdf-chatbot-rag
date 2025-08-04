from typing import List, Dict, Any
import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.config import Config

class EmbeddingManager:
    """
    Manages embeddings and retrieval using LangChain components.
    Uses SentenceTransformerEmbeddings for embeddings and FAISS for vector storage.
    """
    def __init__(self):
        # Initialize the embedding model using LangChain's HuggingFaceEmbeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.vectorstore = None
        self.retriever = None

    def create_embeddings(self, documents: List[Document]):
        """
        Creates embeddings for documents and stores them in FAISS.
        
        Args:
            documents: List of LangChain Document objects
        """
        try:
            # Create FAISS index from documents using LangChain
            self.vectorstore = FAISS.from_documents(
                documents, 
                self.embedding_model
            )
            
            # Create a retriever from the vector store
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": Config.TOP_K}
            )
            
            return True
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            return False

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Searches for relevant documents based on the query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve (defaults to Config.TOP_K)
            
        Returns:
            List[Document]: A list of relevant Document objects
        """
        if not k:
            k = Config.TOP_K
            
        if not self.vectorstore:
            return []
            
        try:
            # Use the retriever to get relevant documents
            relevant_docs = self.retriever.get_relevant_documents(query)
            return relevant_docs
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []