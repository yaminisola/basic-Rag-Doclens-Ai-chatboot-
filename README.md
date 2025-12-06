# 📚 DocLens AI Chatbot

An intelligent document chatbot powered by RAG (Retrieval-Augmented Generation) technology that allows users to upload documents and have natural conversations about their content.

## 🌟 Features

- **Document Upload & Processing**: Upload PDF, TXT, and DOCX files for analysis
- **Intelligent Question Answering**: Ask questions about your documents and get accurate, context-aware responses
- **Source Citation**: Responses include references to specific parts of the documents
- **Vector-based Retrieval**: Uses embeddings for semantic search and retrieval
- **Conversational Interface**: Natural chat experience with document understanding
- **Multi-document Support**: Process and query multiple documents simultaneously

## 🚀 Getting Started

<img width="1875" height="950" alt="image" src="https://github.com/user-attachments/assets/5c6b44c5-d84f-44e2-8e7d-68f7bd61c66e" />


### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/doclens-ai.git
cd doclens-ai
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` file and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
# or
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Running the Application

```bash
streamlit run app.py
```

or

```bash
python app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
RAG-PROJECT/
├── __pycache__/          # Python cache files
├── .streamlit/           # Streamlit configuration
├── .venv/                # Virtual environment
├── uploads/              # Uploaded documents storage
├── vectorstore/          # Vector database storage
├── .env                  # Environment variables (create this)
├── app.py                # Main application file
├── embeddings.py         # Embedding generation logic
├── ingest.py             # Document ingestion pipeline
├── rag.py                # RAG implementation
├── requirements.txt      # Python dependencies
├── utils.py              # Utility functions
└── README.md             # This file
```

## 🔧 Configuration

### Vector Store
The application uses a local vector store to store document embeddings. The default location is `./vectorstore/`.

### Supported File Types
- PDF (`.pdf`)
- Text files (`.txt`)
- Word documents (`.docx`)

### File Size Limit
Maximum file size: 200MB per file

## 💡 Usage

1. **Upload a Document**
   - Click "Browse files" or drag and drop your document
   - Wait for processing to complete

2. **Process Document**
   - Click "Process Document" button
   - The system will chunk and embed your document

3. **Ask Questions**
   - Type your question in the chat input
   - Get AI-powered responses based on document content
   - Review source citations for transparency

4. **Clear History**
   - Use "Clear Chat History" to start a new conversation

## 🛠️ Technologies Used

- **LangChain**: Framework for LLM applications
- **OpenAI / Anthropic**: Language models for generation
- **FAISS / Chroma**: Vector database for embeddings
- **Streamlit**: Web interface framework
- **Python**: Core programming language

## 📊 How It Works

1. **Document Ingestion**: Documents are uploaded and split into chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in a vector database
4. **Query Processing**: User questions are embedded and used to retrieve relevant chunks
5. **Response Generation**: Retrieved context is used to generate accurate answers


## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced document preprocessing
- [ ] Conversation history persistence
- [ ] Document comparison features
- [ ] Export chat history
- [ ] Mobile-responsive design
- [ ] User authentication
- [ ] Cloud deployment ready

