from typing import List, Dict, Any, Optional
import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.config import Config

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

class EmbeddingManager:
    """
    Manages embeddings and retrieval using LangChain components.
    Uses SentenceTransformerEmbeddings for embeddings and FAISS for vector storage.
    """
    def __init__(self):
        # Defer embedding model creation to avoid heavy imports / device moves at app import time
        self.embedding_model: Optional[Any] = None
        self.vectorstore = None
        self.retriever = None

    def _init_embedding_model(self):
        """
        Lazily initialize the embedding model. Try to use SentenceTransformer on CPU.
        If that fails (e.g., meta tensor -> NotImplementedError), fall back to a
        lightweight TF-IDF based embedding implementation so the app remains usable.
        """
        if self.embedding_model is not None:
            return

        try:
            self.embedding_model = SentenceTransformerEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
            return
        except Exception as e:
            print(f"SentenceTransformer init failed, falling back to TF-IDF embeddings: {e}")

        if TfidfVectorizer is None:
            raise RuntimeError(
                "No viable embedding backend available and scikit-learn is not installed."
            )

        class TfidfEmbeddings:
            """Minimal embedding adapter providing embed_documents and embed_query."""
            def __init__(self):
                self.vectorizer = TfidfVectorizer()
                self._fitted = False

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                if not self._fitted:
                    X = self.vectorizer.fit_transform(texts)
                    self._fitted = True
                else:
                    X = self.vectorizer.transform(texts)
                return [row.toarray().ravel().tolist() for row in X]

            def embed_query(self, text: str) -> List[float]:
                if not self._fitted:
                    X = self.vectorizer.fit_transform([text])
                    self._fitted = True
                else:
                    X = self.vectorizer.transform([text])
                return X.toarray().ravel().tolist()

        self.embedding_model = TfidfEmbeddings()

    def create_embeddings(self, documents: List[Document]):
        """
        Creates embeddings for documents and stores them in FAISS.
        
        Args:
            documents: List of LangChain Document objects
        """
        try:
            if not documents:
                print("No documents provided to create_embeddings(). Skipping.")
                return False
            # Ensure embedding model is ready (lazy init)
            self._init_embedding_model()
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
            if not self.retriever and self.vectorstore:
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k or Config.TOP_K}
                )

            relevant_docs = self.retriever.get_relevant_documents(query)
            return relevant_docs
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

    def clear_embeddings(self):
        """Clear the in-memory vectorstore and retriever."""
        self.vectorstore = None
        self.retriever = None