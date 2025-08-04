import streamlit as st
import time
from src.processor import PDFProcessor
from src.embedding import EmbeddingManager
from src.chat import ChatManager
from src.config import Config
from langchain.schema import Document

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

def initialize_session_state():
    """
    Initialize the Streamlit session state with the required components.
    Creates instances of the processor, embedding manager, and chat manager.
    """
    if "processor" not in st.session_state:
        st.session_state.processor = PDFProcessor()
        
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
        
    if "chat_manager" not in st.session_state:
        # Check if API key is available
        if not Config.is_valid():
            st.error("Missing API key. Please check your .env file.")
            return False
            
        st.session_state.chat_manager = ChatManager(Config.GOOGLE_API_KEY)
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "documents" not in st.session_state:
        st.session_state.documents = []
        
    return True

def process_documents(uploaded_files):
    """
    Process the uploaded PDF documents.
    
    Args:
        uploaded_files: List of uploaded PDF files
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        with st.spinner("Processing documents..."):
            all_documents = []
            
            for file in uploaded_files:
                # Process each document using LangChain pipeline
                documents = st.session_state.processor.process_document(file)
                all_documents.extend(documents)
                
            # Store processed documents in session state
            st.session_state.documents = all_documents
            
            # Create embeddings for the documents
            success = st.session_state.embedding_manager.create_embeddings(all_documents)
            
            if success:
                # Connect the retriever to the chat manager
                st.session_state.chat_manager.set_retriever(
                    st.session_state.embedding_manager.retriever
                )
                st.success(f"Successfully processed {len(all_documents)} document chunks!")
                return True
            else:
                st.error("Failed to create embeddings.")
                return False
                
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def main():
    """
    Main application function.
    Sets up the UI and handles user interactions.
    """
    # Initialize session state
    if not initialize_session_state():
        return
    
    st.title("📚 PDF Chat Assistant")
    
    # Sidebar for document upload and controls
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            process_button = st.button("Process Documents")
            if process_button:
                process_documents(uploaded_files)
                
        if st.session_state.documents:
            st.success(f"{len(st.session_state.documents)} chunks in memory")
            
            # Add a button to clear the conversation history
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.session_state.chat_manager.reset_conversation()
                st.experimental_rerun()
            # Add a button to clear file chunks
            if st.button("Clear File Chunks"):
                st.session_state.documents = []
                if hasattr(st.session_state.embedding_manager, 'clear_embeddings'):
                    st.session_state.embedding_manager.clear_embeddings()
                st.experimental_rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if query := st.chat_input("Ask your question"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Check if documents have been uploaded and processed
        if not st.session_state.documents:
            with st.chat_message("assistant"):
                st.write("Please upload and process PDF documents first!")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload and process PDF documents first!"})
            return

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # If using direct retriever-based approach
                if hasattr(st.session_state.embedding_manager, 'retriever') and st.session_state.embedding_manager.retriever:
                    # Using LangChain's conversational retrieval chain
                    response = st.session_state.chat_manager.generate_response(query, [])
                else:
                    # Fallback to manual retrieval and response generation
                    relevant_docs = st.session_state.embedding_manager.search(query)
                    response = st.session_state.chat_manager.generate_response(query, relevant_docs)
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()