"""
config.py
Centralized configuration for RecruitRadar AI pipeline.
All tunable parameters live here — no magic numbers in the codebase.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """Configuration for the RAG retrieval pipeline."""
    
    # Embedding model — runs locally, no API cost
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking strategy
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Retrieval
    top_k_chunks: int = 5
    similarity_metric: str = "cosine"
    
    # Vector store collection name
    collection_name: str = "resume_chunks"


@dataclass
class LLMConfig:
    """Configuration for the LLM scoring agent."""
    
    # Groq API
    api_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = "llama-3.3-70b-versatile"
    
    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 1000
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class EvalConfig:
    """Configuration for pipeline quality evaluation."""
    
    # Scoring model — same as LLM for zero extra cost
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 5
    
    # Score thresholds
    strong_match_threshold: int = 75
    moderate_match_threshold: int = 50
    
    # Retrieval quality thresholds
    good_retrieval_score: float = 7.0
    good_coverage_score: float = 7.0


@dataclass
class AppConfig:
    """Top-level application configuration."""
    
    rag: RAGConfig = None
    llm: LLMConfig = None
    eval: EvalConfig = None
    
    # File paths
    resume_path: str = "resume.pdf"
    
    def __post_init__(self):
        self.rag = RAGConfig()
        self.llm = LLMConfig()
        self.eval = EvalConfig()


# Global config instance
config = AppConfig()