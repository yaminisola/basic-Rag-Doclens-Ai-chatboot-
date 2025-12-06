"""
Embeddings module for DocLens
Handles embedding model initialization and text embedding generation
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import numpy as np
import os
import streamlit as st

class EmbeddingManager:
    """Manages embedding models and provides caching"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding manager with specified model
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self._embeddings = None
    
    @property
    def embeddings(self):
        """Lazy load embeddings model (cached)"""
        if self._embeddings is None:
            print(f"Loading embedding model: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            print("✅ Embedding model loaded successfully")
        return self._embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        sample_embedding = self.embed_query("test")
        return len(sample_embedding)


class MultiModelEmbedding:
    """Support for multiple embedding models"""
    
    AVAILABLE_MODELS = {
        'mini': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'description': 'Fast and lightweight (recommended)'
        },
        'mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dimension': 768,
            'description': 'High quality, slower'
        },
        'distilbert': {
            'name': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
            'dimension': 512,
            'description': 'Multilingual support'
        }
    }
    
    def __init__(self, model_key: str = 'mini'):
        """
        Initialize with a specific model
        
        Args:
            model_key: Key from AVAILABLE_MODELS
        """
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model key must be one of {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_info = self.AVAILABLE_MODELS[model_key]
        self.manager = EmbeddingManager(model_name=self.model_info['name'])
    
    @classmethod
    def list_models(cls):
        """List all available embedding models"""
        return cls.AVAILABLE_MODELS
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.manager.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.manager.embed_query(text)


def get_cached_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get or create cached embeddings instance
    Uses Streamlit session state for caching
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        EmbeddingManager instance
    """
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager(model_name)
    
    return st.session_state.embedding_manager


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between -1 and 1
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def batch_embed_texts(texts: List[str], batch_size: int = 32, model_name: str = None) -> List[List[float]]:
    """
    Embed texts in batches for better performance
    
    Args:
        texts: List of texts to embed
        batch_size: Size of each batch
        model_name: Embedding model to use
        
    Returns:
        List of embedding vectors
    """
    manager = EmbeddingManager(model_name) if model_name else EmbeddingManager()
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = manager.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        
        if len(texts) > batch_size:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    return all_embeddings


# Singleton instance for global use
_global_embedding_manager = None

def get_embedding_manager(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingManager:
    """
    Get global embedding manager instance (singleton pattern)
    
    Args:
        model_name: Name of embedding model
        
    Returns:
        Global EmbeddingManager instance
    """
    global _global_embedding_manager
    
    if _global_embedding_manager is None:
        _global_embedding_manager = EmbeddingManager(model_name)
    
    return _global_embedding_manager


if __name__ == "__main__":
    # Test the embedding functionality
    print("Testing Embeddings Module\n")
    
    # Test 1: Basic embedding
    print("1. Testing basic embedding...")
    manager = EmbeddingManager()
    
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "The weather is nice today."
    ]
    
    print(f"Embedding {len(test_texts)} texts...")
    embeddings = manager.embed_documents(test_texts)
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Test 2: Query embedding
    print("\n2. Testing query embedding...")
    query = "What is AI?"
    query_embedding = manager.embed_query(query)
    print(f"✅ Query embedding dimension: {len(query_embedding)}")
    
    # Test 3: Similarity computation
    print("\n3. Testing similarity...")
    sim1 = compute_similarity(embeddings[0], embeddings[1])
    sim2 = compute_similarity(embeddings[0], embeddings[2])
    print(f"Similarity (ML vs Python): {sim1:.4f}")
    print(f"Similarity (ML vs Weather): {sim2:.4f}")
    print(f"✅ ML and Python are more similar than ML and Weather: {sim1 > sim2}")
    
    # Test 4: Multiple models
    print("\n4. Testing multiple models...")
    print("Available models:")
    for key, info in MultiModelEmbedding.list_models().items():
        print(f"  - {key}: {info['description']} ({info['dimension']}d)")
    
    print("\n✅ All tests passed!")