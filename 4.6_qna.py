import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
import chainlit as cl

# Load environment variables
load_dotenv()

# Azure Document Intelligence Setup
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# Initialize the LLM with streaming enabled
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=os.getenv("OPEN_AI_API_KEY"), streaming=True)

embedding_function = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002"
)

# Query decomposition prompt template
decomposition_template = """
You are an AI assistant tasked with breaking down complex questions into simpler sub-questions.
Given the following question, please break it down into 2-4 simpler sub-questions that, when answered, would help address the main question.

Main question: {question}

Sub-questions:
1.
2.
3. (if necessary)
4. (if necessary)
"""
decomposition_prompt = PromptTemplate(
    template=decomposition_template,
    input_variables=["question"]
)

# Function to decompose query into sub-questions
async def decompose_query(question: str) -> List[str]:
    decomposition_chain = LLMChain(llm=llm, prompt=decomposition_prompt)
    result = await decomposition_chain.arun(question)
    sub_questions = [q.strip() for q in result.split('\n') if q.strip() and not q.strip().startswith('Sub-questions:')]
    return sub_questions

# Function to answer each sub-question
async def answer_question(chain, question: str) -> Dict[str, Any]:
    return await chain.acall(question)

# Add a new prompt template for generating suggested questions
suggested_questions_template = """
Based on the following context and the user's previous question, generate 3 relevant follow-up questions that the user might be interested in asking next:

Context: {context}
Previous question: {question}

Suggested questions:
1.
2.
3.
"""
suggested_questions_prompt = PromptTemplate(
    template=suggested_questions_template,
    input_variables=["context", "question"]
)
# Function to generate suggested questions
async def generate_suggested_questions(chain, question: str, context: str) -> List[str]:
    suggested_questions_chain = LLMChain(llm=llm, prompt=suggested_questions_prompt)
    result = await suggested_questions_chain.arun(context=context, question=question)
    suggested_questions = [q.strip() for q in result.split('\n') if q.strip() and q.strip()[0].isdigit()]
    return suggested_questions

@cl.on_chat_start
async def start():
    # Request file upload at the start
    files = await cl.AskFileMessage(
        content="Please upload PDF files to begin!",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180,
        max_files=5,
    ).send()

    msg = cl.Message(content=f"Processing {len(files)} PDF files...")
    await msg.send()

    # Process the uploaded PDFs using AzureAIDocumentIntelligenceLoader
    documents = []
    for file in files:
        loader = AzureAIDocumentIntelligenceLoader(
            file_path=file.path,
            api_key=doc_intelligence_key,
            api_endpoint=doc_intelligence_endpoint,
            api_model="prebuilt-layout",
        )
        documents.extend(await cl.make_async(loader.load)())

    # Split documents into sections based on headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    texts = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            texts.append(Document(page_content=split.page_content, metadata={**doc.metadata, **split.metadata}))

    # Ensure each text has metadata for source and page
    for text in texts:
        text.metadata["source"] = f"{text.metadata.get('source', 'HR-PDF')}"
        text.metadata["page"] = text.metadata.get("page_number", "unknown")

    # Build FAISS index from the documents for semantic search
    docsearch = await cl.make_async(FAISS.from_documents)(texts, embedding_function)

    # Initialize BM25 and Semantic retrievers
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 5
    semantic_retriever = docsearch.as_retriever(search_kwargs={"k": 5})

    # Use EnsembleRetriever to combine BM25 and semantic retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5]
    )

    # Setup conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Initialize ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        memory=memory,
        return_source_documents=True,
    )

    msg.content = "Processing complete. You can now ask questions!"
    await msg.update()

    # Store the chain in user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    # Step 1: Decompose the query into sub-questions
    msg = cl.Message(content="Decomposing the query into sub-questions...")
    await msg.send()
    
    sub_questions = await decompose_query(message.content)
    msg.content = f"Sub-questions: {', '.join(sub_questions)}"
    await msg.update()

    # Step 2: Perform hybrid search on each sub-question
    msg.content = "Performing hybrid search on sub-questions..."
    await msg.update()

    tasks = [answer_question(chain, q) for q in sub_questions]
    sub_answers = await asyncio.gather(*tasks)

    # Step 3: Combine sub-answers into one response
    msg.content = "Combining sub-answers..."
    await msg.update()

    combined_answer = "\n\n".join([f"Q: {q}\nA: {a['answer']}" for q, a in zip(sub_questions, sub_answers)])

    # Step 4: Generate the final answer based on sub-answers
    msg.content = "Generating the final answer..."
    await msg.update()

    final_answer_prompt = PromptTemplate(
        template="Based on the following sub-questions and answers, provide a comprehensive answer to the main question: {main_question}\n\nSub-questions and answers:\n{combined_answer}\n\nFinal answer:",
        input_variables=["main_question", "combined_answer"]
    )
    final_answer_chain = LLMChain(llm=llm, prompt=final_answer_prompt)
    final_answer = await final_answer_chain.arun(main_question=message.content, combined_answer=combined_answer)




    # Step 5: Handle source documents for citation
    all_source_documents = []
    for sub_answer in sub_answers:
        all_source_documents.extend(sub_answer.get("source_documents", []))

    unique_sources = {}
    for doc in all_source_documents:
        source_key = (doc.metadata['source'], doc.metadata['page'])
        if source_key not in unique_sources:
            unique_sources[source_key] = doc

    text_elements = []
    if unique_sources:
        for idx, (source_key, doc) in enumerate(unique_sources.items()):
            source_content = f"Source: {source_key[0]}, Page: {source_key[1]}\n\n{doc.page_content[:500]}..."  # Truncate content for brevity
            text_elements.append(cl.Text(content=source_content, name=f"source_{idx+1}", display="side"))

        source_citations = [f"[{idx + 1}] {source_key[0]} (p. {source_key[1]})" for idx, source_key in enumerate(unique_sources.keys())]
        final_answer += f"\n\nSources: {', '.join(source_citations)}"
    else:
        final_answer += "\nNo sources found."

    # Send the final answer with sources
    await cl.Message(content=final_answer, elements=text_elements).send()

        # After sending the final answer, generate suggested questions
    context = final_answer  # Use the final answer as context for generating suggestions
    suggested_questions = await generate_suggested_questions(chain, message.content, context)

    # Create suggestion elements for Chainlit
    elements = [
        cl.Text(content=question, title=f"Suggestion {i+1}")
        for i, question in enumerate(suggested_questions)
    ]
        # Send suggestions to the user
    await cl.Message(
        content="Here are some suggested follow-up questions:",
        elements=elements
    ).send()
