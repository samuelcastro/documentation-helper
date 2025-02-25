# LangChain Documentation Helper

A Streamlit-based chatbot that helps you navigate and understand the LangChain documentation. This application uses LangChain, OpenAI embeddings, and Pinecone vector store to provide accurate and contextual responses to your questions about LangChain.

## Features

- Interactive chat interface built with Streamlit
- Semantic search powered by OpenAI embeddings and Pinecone vector store
- Source citations for every response
- Context-aware responses using LangChain's retrieval chain

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment
- LangChain documentation downloaded locally

## Setup

1. Clone the repository

2. Install dependencies:
```bash
pip install langchain langchain-openai langchain-pinecone python-dotenv streamlit
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

4. Download the LangChain documentation:
```bash
wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/latest/
```

## Usage

1. First, ingest the documentation:
```bash
python ingestion.py
```
This will:
- Load the documentation
- Split it into semantic chunks
- Create embeddings using OpenAI
- Store the embeddings in Pinecone

2. Run the Streamlit application:
```bash
streamlit run main.py
```

3. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

## Project Structure

- `main.py`: Streamlit application interface
- `ingestion.py`: Documentation ingestion and embedding creation
- `backend/core.py`: Core functionality for the LLM chain
- `.env`: Environment variables configuration

## How It Works

1. The application uses `ReadTheDocsLoader` to load the LangChain documentation
2. Documents are split into semantic chunks using `RecursiveCharacterTextSplitter`
3. OpenAI embeddings are created for each chunk and stored in Pinecone
4. When a query is made, the application:
   - Searches for relevant documentation chunks using similarity search
   - Combines the chunks with the query using LangChain's retrieval chain
   - Generates a response using OpenAI's chat model
   - Provides source citations for verification
