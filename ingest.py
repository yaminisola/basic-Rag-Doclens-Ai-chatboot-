import os
import pickle
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores.faiss import FAISS
import streamlit as st
from embeddings import get_embedding_manager

# Directory for storing vectorstore
VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

def load_document(file_path: str):
    """Load document based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        documents = loader.load()
        return documents
    
    except Exception as e:
        print(f"Error loading document: {e}")
        return None

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks for better retrieval
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)
    
    return chunks

def get_embeddings():
    """
    Get embeddings instance using the embeddings module
    Uses global singleton for efficiency
    """
    return get_embedding_manager().embeddings

def load_existing_vectorstore():
    """Load existing vectorstore if it exists"""
    try:
        if os.path.exists(VECTORSTORE_PATH):
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded existing vectorstore from {VECTORSTORE_PATH}")
            return vectorstore
        return None
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def save_vectorstore(vectorstore):
    """Save vectorstore to disk"""
    try:
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"✅ Vectorstore saved to {VECTORSTORE_PATH}")
        return True
    except Exception as e:
        print(f"Error saving vectorstore: {e}")
        return False

def process_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
    """
    Process a document: load, chunk, embed, and store in vectorstore
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Step 1: Load document
        print(f"📄 Loading document: {file_path}")
        documents = load_document(file_path)
        
        if not documents:
            print("❌ Failed to load document")
            return False
        
        print(f"✅ Loaded {len(documents)} pages/sections")
        
        # Step 2: Split into chunks
        print(f"✂️ Splitting document into chunks (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Add source file metadata to all chunks
        for chunk in chunks:
            chunk.metadata['source_file'] = os.path.basename(file_path)
            chunk.metadata['file_path'] = file_path
        
        # Step 3: Get embeddings model
        print("🧠 Loading embedding model...")
        embeddings = get_embeddings()
        
        # Step 4: Load existing vectorstore or create new one
        print("💾 Processing vectorstore...")
        existing_vectorstore = load_existing_vectorstore()
        
        if existing_vectorstore:
            # Add new documents to existing vectorstore
            print("📥 Adding to existing vectorstore...")
            existing_vectorstore.add_documents(chunks)
            vectorstore = existing_vectorstore
            print(f"✅ Added {len(chunks)} new chunks to existing vectorstore")
        else:
            # Create new vectorstore
            print("🆕 Creating new vectorstore...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            print(f"✅ Created new vectorstore with {len(chunks)} chunks")
        
        # Step 5: Save vectorstore
        print("💾 Saving vectorstore...")
        if save_vectorstore(vectorstore):
            # Save metadata about processed document
            metadata_path = os.path.join(VECTORSTORE_DIR, "documents_metadata.pkl")
            
            # Load existing metadata or create new
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    all_metadata = pickle.load(f)
            else:
                all_metadata = []
            
            # Add current document metadata
            doc_metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'num_pages': len(documents),
                'num_chunks': len(chunks),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'timestamp': str(os.path.getmtime(file_path))
            }
            all_metadata.append(doc_metadata)
            
            # Save updated metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(all_metadata, f)
            
            print("✅ Document processed successfully!")
            print(f"📊 Total chunks in vectorstore: {vectorstore.index.ntotal}")
            return True
        else:
            print("❌ Failed to save vectorstore")
            return False
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_vectorstore():
    """Get the current vectorstore"""
    return load_existing_vectorstore()

def get_document_stats():
    """Get statistics about processed documents"""
    try:
        metadata_path = os.path.join(VECTORSTORE_DIR, "documents_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        return []
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return []

def clear_vectorstore():
    """Clear all stored vectors and metadata"""
    try:
        if os.path.exists(VECTORSTORE_DIR):
            import shutil
            shutil.rmtree(VECTORSTORE_DIR)
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            print("✅ Vectorstore cleared")
        return True
    except Exception as e:
        print(f"❌ Error clearing vectorstore: {e}")
        return False

def search_similar_chunks(query: str, top_k: int = 5):
    """
    Search for similar chunks in the vectorstore
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List of (document, score) tuples
    """
    try:
        vectorstore = get_vectorstore()
        if not vectorstore:
            return []
        
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        return results
    except Exception as e:
        print(f"Error searching: {e}")
        return []

if __name__ == "__main__":
    # Test the ingestion process
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        print("=" * 60)
        print("DocLens Document Ingestion Test")
        print("=" * 60)
        
        success = process_document(file_path)
        
        print("\n" + "=" * 60)
        if success:
            print("✅ Processing SUCCESSFUL")
            
            # Show statistics
            stats = get_document_stats()
            print(f"\n📊 Total documents processed: {len(stats)}")
            
            vectorstore = get_vectorstore()
            if vectorstore:
                print(f"📊 Total chunks in database: {vectorstore.index.ntotal}")
        else:
            print("❌ Processing FAILED")
        print("=" * 60)
    else:
        print("Usage: python ingest.py <file_path>")
        print("\nExample:")
        print("  python ingest.py document.pdf")