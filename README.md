# Chainlit App with Query Decomposition and Hybrid Search

This project is a Chainlit application built using LangChain, incorporating features such as query decomposition and hybrid search. It enables users to upload PDF documents, break down complex queries into simpler sub-questions, and fetch answers using a combination of semantic search and BM25 retrieval techniques.

## Features
- **Query Decomposition**: Automatically breaks down complex queries into simpler sub-questions.
- **Hybrid Search**: Combines semantic search using vector embeddings (Chroma) and keyword search (BM25) for efficient document retrieval.
- **Azure Document Intelligence**: Processes PDFs using Azure AI Document Intelligence to extract meaningful chunks for better searchability.
- **Conversational Interface**: Built on Chainlit for an interactive chat-based user experience.
- **Memory-Powered Conversations**: Utilizes LangChain's memory capabilities to maintain chat history and improve conversational context over time.

## Project Structure
📁 your-repo/ │ ├── 📄 .gitignore # Specifies untracked files to be ignored by Git ├── 📄 README.md # This README file ├── 📄 pyproject.toml # Poetry configuration file for managing dependencies ├── 📄 chainlit_config.py # Chainlit app configuration file └── 📁 src/ # Source code directory ├── 📄 main.py # Main application logic └── 📁 assets/ # Assets like PDF files and logs (if applicable)
