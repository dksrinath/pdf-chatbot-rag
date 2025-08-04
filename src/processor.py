import io
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.config import Config
import PyPDF2

class PDFProcessor:
    """
    Processes PDF documents using LangChain components for document loading and text splitting.
    Uses PyPDFLoader for document loading and RecursiveCharacterTextSplitter for text chunking.
    """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )

    def extract_text(self, pdf_file) -> str:
        """
        Extracts text from a PDF file using PyPDF2.
        
        Args:
            pdf_file: A file-like object (from Streamlit's file_uploader)
            
        Returns:
            str: The extracted text from the PDF
        """
        # Use PyPDF2 directly since PyPDFLoader doesn't support file objects easily
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks using LangChain's RecursiveCharacterTextSplitter.
        
        Args:
            text: The text to split
            
        Returns:
            List[str]: A list of text chunks
        """
        return self.text_splitter.split_text(text)
        
    def process_document(self, pdf_file) -> List[Document]:
        """
        Full document processing pipeline: extract text and split into chunks.
        
        Args:
            pdf_file: A file-like object (from Streamlit's file_uploader)
            
        Returns:
            List[Document]: A list of LangChain Document objects with text chunks
        """
        try:
            text = self.extract_text(pdf_file)
            chunks = self.split_text(text)
            
            # Convert to LangChain Document objects with metadata
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_file.name,
                        "chunk_id": i
                    }
                ) for i, chunk in enumerate(chunks)
            ]
            
            return documents
        except Exception as e:
            print(f"Error processing document {pdf_file.name}: {str(e)}")
            return []