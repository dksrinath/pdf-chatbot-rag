import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """
    Configuration class for the application.
    Centralizes all configuration parameters and supports environment variable overrides.
    """
    # API Keys
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Text chunking parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Retrieval settings
    TOP_K = int(os.getenv("TOP_K", "4"))
    
    # LLM parameters
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Memory settings
    MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))
    
    @classmethod
    def get_llm_params(cls) -> Dict[str, Any]:
        """
        Get LLM parameters as a dictionary for easy configuration.
        
        Returns:
            Dict[str, Any]: Dictionary of LLM parameters
        """
        return {
            "temperature": cls.LLM_TEMPERATURE,
            "max_output_tokens": cls.MAX_TOKENS,
        }
    
    @classmethod
    def is_valid(cls) -> bool:
        """
        Check if the configuration is valid (has required API keys).
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """

        return bool(cls.GOOGLE_API_KEY)
