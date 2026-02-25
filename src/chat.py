from typing import List, Dict, Any, Optional
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.config import Config

class ChatManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.memory = None
        self.chain = None
        self.llm = None
        self._initialize_components()
        
    def _initialize_components(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
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
            self.llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it", 
                google_api_key=self.api_key
            )
    
    def _create_chain(self, retriever):
        system_template = """You are a helpful assistant that answers questions based on the provided context.
        If you cannot find the answer in the context, acknowledge that and provide general information if possible.
        Always cite your sources when the information comes from the provided context.
        
        Context:
        {context}
        """
        
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_template + "\nQuestion: {question}"
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )
        return self.chain
    
    def _fallback_generation(self, query: str, context_text: str):
        """Helper to generate response without the chain if chain fails."""
        try:
            messages = [
                SystemMessage(content=f"You are a helpful assistant... Context:\n{context_text}"),
                HumanMessage(content=query),
            ]
            # Retry logic for the fallback itself
            for attempt in range(2):
                try:
                    response = self.llm.invoke(messages)
                    return {"answer": response.content, "sources": []}
                except Exception:
                    time.sleep(2)
            return {"answer": "I am currently overloaded. Please try again in a few seconds.", "sources": []}
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": []}

    def generate_response(self, query: str, context_docs: List[Document]):
        # Rate limit protection (Gemma 3 allows 30 RPM, so 2s is safe)
        time.sleep(2) 

        context_texts = [doc.page_content for doc in context_docs]
        context_text = "\n".join(context_texts)

        # 1. Try to use the RAG Chain first
        if self.chain:
            try:
                result = self.chain.invoke({"question": query})
                answer = result.get("answer") or result.get("output_text") or ""
                sources = []
                for d in result.get("source_documents", []):
                    md = getattr(d, 'metadata', {})
                    sources.append({
                        "text": d.page_content,
                        "metadata": md,
                    })
                return {"answer": answer, "sources": sources}
            except Exception as e:
                print(f"Chain error: {str(e)}")
                # DO NOT RECURSE. Fallback explicitly.
                pass 
        
        # 2. Fallback to direct generation if chain missing or failed
        print("Falling back to direct generation...")
        return self._fallback_generation(query, context_text)
                
    def set_retriever(self, retriever):
        self._create_chain(retriever)
        
    def reset_conversation(self):
        if self.memory:
            self.memory.clear()
            
    def get_conversation_history(self):
        if self.memory:
            return self.memory.chat_memory.messages
        return []