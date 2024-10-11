import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import chainlit as cl

# Load environment variables
load_dotenv()

# Setting up Azure Document Intelligence
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0,
    streaming=True
)

embedding_function = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002"
)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload PDF files
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload PDF files to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
            max_files=5,
        ).send()

    msg = cl.Message(content=f"Processing {len(files)} PDF files...")
    await msg.send()

    documents = []

    for file in files:
        loader = AzureAIDocumentIntelligenceLoader(
            file_path=file.path,
            api_key=doc_intelligence_key,
            api_endpoint=doc_intelligence_endpoint,
            api_model="prebuilt-layout",
        )
        documents.extend(await cl.make_async(loader.load)())

    # Split the documents into chunks based on markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Process each document separately
    texts = []
    for doc in documents:
        # Split the document content
        splits = markdown_splitter.split_text(doc.page_content)
        # Create new Document objects with split content and original metadata
        for split in splits:
            texts.append(Document(page_content=split.page_content, metadata={**doc.metadata, **split.metadata}))

    # Create metadata for each chunk
    for i, text in enumerate(texts):
        text.metadata["source"] = f"{text.metadata.get('source', 'unknown')}-{i}"

    # Create a Chroma vector store
    docsearch = await cl.make_async(Chroma.from_documents)(
        texts, embedding_function
    )

    # Create a BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 5  # Set the number of documents to retrieve

    # Create a semantic search retriever
    semantic_retriever = docsearch.as_retriever(search_kwargs={"k": 5})

    # Create an ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5]
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the ensemble retriever
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing {len(files)} PDF files done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name, display="side")
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()