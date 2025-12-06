import streamlit as st
from groq import Groq
import os
from datetime import datetime
from ingest import process_document
from rag import query_documents

# Page configuration
st.set_page_config(
    page_title="DocLens AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize Groq client
if "groq_client" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if api_key:
        st.session_state.groq_client = Groq(api_key=api_key)
    else:
        st.error("Please set GROQ_API_KEY in your environment or secrets")
        st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# Sidebar for document upload
with st.sidebar:
    st.title("📁 Document Manager")
    st.write("Upload documents to chat about them!")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Upload PDF, TXT, or DOCX files"
    )
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file temporarily
                    file_path = os.path.join("uploads", uploaded_file.name)
                    os.makedirs("uploads", exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the document
                    success = process_document(file_path)
                    
                    if success:
                        st.session_state.processed_docs.append(uploaded_file.name)
                        st.session_state.vectorstore_ready = True
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                        
                        # Add system message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"📄 I've processed **{uploaded_file.name}**. You can now ask me questions about it!",
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                    else:
                        st.error("Failed to process document")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Show processed documents
    if st.session_state.processed_docs:
        st.divider()
        st.subheader("📚 Processed Documents")
        for doc in st.session_state.processed_docs:
            st.write(f"✓ {doc}")
    
    # Clear chat button
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🤖 DocLens AI Chatbot")
st.caption("Chat with your documents or ask general questions!")

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents or general questions..."):
    # Add user message to chat
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Check if we should use RAG (document-based answer)
            use_rag = False
            if st.session_state.vectorstore_ready:
                # Keywords that suggest document-specific questions
                doc_keywords = [
                    "document", "pdf", "file", "uploaded", "according to",
                    "what does", "summarize", "tell me about", "explain",
                    "find", "search", "show me"
                ]
                use_rag = any(keyword in prompt.lower() for keyword in doc_keywords)
            
            if use_rag:
                # Use RAG for document-specific questions
                with st.spinner("Searching through documents..."):
                    response_text = query_documents(prompt, st.session_state.groq_client)
                    message_placeholder.markdown(response_text)
            else:
                # Use general chat for other questions
                with st.spinner("Thinking..."):
                    # Prepare conversation history
                    messages_for_api = [
                        {"role": "system", "content": "You are a helpful AI assistant. You can answer general questions and help with document analysis when needed."}
                    ]
                    
                    # Add recent chat history (last 10 messages)
                    for msg in st.session_state.messages[-10:]:
                        if msg["role"] in ["user", "assistant"]:
                            messages_for_api.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                    
                    # Call Groq API with UPDATED MODEL
                    response = st.session_state.groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",  # FIXED: Updated model name
                        messages=messages_for_api,
                        temperature=0.7,
                        max_tokens=1024,
                        stream=True
                    )
                    
                    # Stream the response
                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    response_text = full_response
            
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })

# Footer
st.divider()
st.caption("💡 Tip: Upload documents in the sidebar to chat about them, or ask me general questions!")