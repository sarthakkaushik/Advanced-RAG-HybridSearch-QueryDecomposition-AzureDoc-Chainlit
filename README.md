# Chainlit App with Query Decomposition and Hybrid Search

This project is a Chainlit application built using LangChain, incorporating features such as query decomposition and hybrid search. It enables users to upload PDF documents, break down complex queries into simpler sub-questions, and fetch answers using a combination of semantic search and BM25 retrieval techniques.

## Features
- **Query Decomposition**: Automatically breaks down complex queries into simpler sub-questions.
- **Hybrid Search**: Combines semantic search using vector embeddings (Chroma) and keyword search (BM25) for efficient document retrieval.
- **Azure Document Intelligence**: Processes PDFs using Azure AI Document Intelligence to extract meaningful chunks for better searchability.
- **Conversational Interface**: Built on Chainlit for an interactive chat-based user experience.
- **Memory-Powered Conversations**: Utilizes LangChain's memory capabilities to maintain chat history and improve conversational context over time.

## Project Structure
📁 your-repo/ │ 
├── 📄 .gitignore # Specifies untracked files to be ignored by Git ├── 📄 README.md # This README file ├── 📄 pyproject.toml # Poetry configuration file for managing dependencies ├── 📄 chainlit_config.py # Chainlit app configuration file └── 📁 src/ # Source code directory ├── 📄 main.py # Main application logic └── 📁 assets/ # Assets like PDF files and logs (if applicable)


## Prerequisites
Before setting up the project, make sure you have the following installed:
- **Python 3.12.x** (The project is tested on Python 3.11.9)
- **Poetry** (for dependency management)

### Install Poetry
If you don't have Poetry installed, you can install it via the command:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```


## Setup
### 1. Clone the Repository
```bash
 git clone https://github.com/your-username/your-repo.git
  cd your-repo
```
### 2. Install Dependencies
Use Poetry to install the project dependencies:
```
poetry install
```
This command reads the pyproject.toml file to install the necessary dependencies, including LangChain, Chainlit, and Azure integrations.

### 3. Set up Environment Variables
Create a .env file in the root directory and add the following environment variables:
```
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_API_VERSION=2023-03-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_ENDPOINT=https://your-openai-endpoint.azure.com
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-document-intelligence-endpoint.azure.com
AZURE_DOCUMENT_INTELLIGENCE_KEY=your_document_intelligence_key
```
### 4. Run the App
Start the Chainlit app with the following command:
```
poetry run chainlit run src/4_qa_qery_decomp_hybrid.py
```

### 5. Upload PDFs and Ask Questions
- After starting the app, you will be prompted to upload PDF files.
- Once the files are processed, you can ask complex questions.
- The app will break the question into simpler sub-questions, search for answers in the documents using both BM25 and semantic search, and provide comprehensive answers with source references.

