from typing import List, Dict, Any, Optional
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from src.config import Config

class ChatManager:
    """
    Manages chat interactions using LangChain components.
    Uses LangChain's ConversationalRetrievalChain for handling conversational RAG.
    """
    def __init__(self, api_key: str):
        """
        Initialize the ChatManager with the specified API key.
        
        Args:
            api_key: The API key for the LLM service
        """
        self.api_key = api_key
        self.memory = None
        self.chain = None
        self.llm = None
        self._initialize_components()
        
    def _initialize_components(self):
        """
        Initialize LangChain components for the chat system.
        Sets up the LLM, memory, and creates the conversation chain.
        """
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize the LLM with retry logic
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=Config.MODEL_NAME,
                google_api_key=self.api_key,
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.95,
                top_k=40
            )
        except Exception as e:
            print(f"Error initializing LLM, retrying: {str(e)}")
            time.sleep(1)
            # Retry once with default parameters
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.api_key
            )
    
    def _create_chain(self, retriever):
        """
        Creates a conversational retrieval chain with the specified retriever.
        
        Args:
            retriever: The document retriever to use
            
        Returns:
            The created conversation chain
        """
        # System prompt template
        system_template = """You are a helpful assistant that answers questions based on the provided context.
        If you cannot find the answer in the context, acknowledge that and provide general information if possible.
        Always cite your sources when the information comes from the provided context.
        
        Context:
        {context}
        """
        
        # Create prompt templates
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_template + "\nQuestion: {question}"
        )
        
        # Create the chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )
        
        return self.chain
    
    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate a response based on the query and retrieved context documents.
        
        Args:
            query: The user's question
            context_docs: List of context documents retrieved for the query
            
        Returns:
            str: The generated response
        """
        # Extract text from documents
        context_texts = [doc.page_content for doc in context_docs]
        context_text = "\n".join(context_texts)
        
        if not self.chain:
            # Handle direct LLM call if chain isn't set up
            try:
                # Format messages with context
                messages = [
                    SystemMessage(content=f"You are a helpful assistant that answers questions based on the provided context. If you cannot find the answer in the context, say so.\n\nContext:\n{context_text}"),
                    HumanMessage(content=query)
                ]
                
                # Call the LLM directly
                response = self.llm(messages)
                return response.content
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                # Implement retry logic
                for attempt in range(3):
                    try:
                        time.sleep(1)  # Wait before retry
                        response = self.llm(messages)
                        return response.content
                    except Exception as retry_e:
                        print(f"Retry {attempt+1} failed: {str(retry_e)}")
                
                return "I encountered an error while processing your request. Please try again later."
        else:
            # Use the chain if available
            try:
                result = self.chain.invoke({"question": query})
                return result["answer"]
            except Exception as e:
                print(f"Chain error: {str(e)}")
                # Fall back to direct LLM call
                return self.generate_response(query, context_docs)
                
    def set_retriever(self, retriever):
        """
        Set the retriever and create a conversation chain.
        
        Args:
            retriever: The document retriever to use
        """
        self._create_chain(retriever)
        
    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        if self.memory:
            self.memory.clear()
            
    def get_conversation_history(self):
        """
        Get the current conversation history.
        
        Returns:
            The conversation history
        """
        if self.memory:
            return self.memory.chat_memory.messages
        return []