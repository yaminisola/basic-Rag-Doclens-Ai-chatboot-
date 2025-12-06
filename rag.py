from ingest import get_vectorstore
from groq import Groq
import os

def query_documents(query: str, groq_client: Groq, top_k: int = 4) -> str:
    """
    Query the vectorstore and generate an answer using RAG
    
    Args:
        query: User's question
        groq_client: Initialized Groq client
        top_k: Number of relevant chunks to retrieve
    
    Returns:
        Generated answer as string
    """
    try:
        # Load vectorstore
        vectorstore = get_vectorstore()
        
        if not vectorstore:
            return "⚠️ No documents have been processed yet. Please upload a document first!"
        
        # Retrieve relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.get_relevant_documents(query)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the uploaded documents. Could you rephrase your question?"
        
        # Combine retrieved context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt for LLM
        system_prompt = """You are a helpful AI assistant that answers questions based on provided document context. 

Instructions:
- Use ONLY the information from the provided context to answer questions
- If the context doesn't contain the answer, say so clearly
- Be concise but comprehensive
- Cite specific parts of the context when relevant
- If asked about specific details, quote relevant parts from the context"""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based on the context provided above. If the answer is not in the context, say that you don't have that information."""

        # Generate response using Groq
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # FIXED: Updated model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        # Add source information
        sources = set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs])
        source_info = "\n\n📚 **Sources:** " + ", ".join([os.path.basename(s) for s in sources])
        
        return answer + source_info
        
    except Exception as e:
        return f"❌ Error querying documents: {str(e)}"

def query_with_conversation_history(
    query: str, 
    groq_client: Groq, 
    conversation_history: list = None,
    top_k: int = 4
) -> str:
    """
    Query documents with conversation history for context-aware responses
    
    Args:
        query: Current user question
        groq_client: Initialized Groq client
        conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        top_k: Number of relevant chunks to retrieve
    
    Returns:
        Generated answer as string
    """
    try:
        vectorstore = get_vectorstore()
        
        if not vectorstore:
            return "⚠️ No documents have been processed yet. Please upload a document first!"
        
        # Retrieve relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.get_relevant_documents(query)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the uploaded documents."
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Build messages with conversation history
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant that answers questions based on document context and conversation history.
                
Instructions:
- Use the provided document context to answer questions
- Consider previous conversation for context
- Be concise and accurate
- If information is not in the documents, say so clearly"""
            }
        ]
        
        # Add conversation history (last 6 messages)
        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"""Document context:
{context}

Question: {query}

Answer based on the document context above."""
        })
        
        # Generate response
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # FIXED: Updated model name
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        # Add sources
        sources = set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs])
        source_info = "\n\n📚 **Sources:** " + ", ".join([os.path.basename(s) for s in sources])
        
        return answer + source_info
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    # Test RAG functionality
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable")
        sys.exit(1)
    
    client = Groq(api_key=api_key)
    
    # Test query
    test_query = "What is this document about?"
    print(f"Query: {test_query}\n")
    
    answer = query_documents(test_query, client)
    print(f"Answer: {answer}")