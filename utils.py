"""
Utility functions for text processing
"""

def chunk_text(text, chunk_size=600, overlap=100):
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text to chunk
        chunk_size: Characters per chunk
        overlap: Characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
    
    return chunks


def clean_text(text):
    """
    Clean extracted text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def count_tokens(text):
    """
    Estimate token count (rough approximation)
    """
    return int(len(text.split()) / 0.75)