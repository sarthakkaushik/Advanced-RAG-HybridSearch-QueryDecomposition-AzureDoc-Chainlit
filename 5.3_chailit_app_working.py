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
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.getenv("OPEN_AI_API_KEY"), streaming=True)

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
# Update the suggested questions template
suggested_questions_template = """
Based on the following context, chat history, and relevant document chunks, generate 3 diverse and relevant follow-up questions that can be answered using the information in our knowledge base:

Context: {context}
Chat History: {chat_history}
Relevant Document Chunks:
{relevant_chunks}

Previous question: {question}

Ensure the suggested questions are diverse, insightful, and can be answered using the information in our document collection. Focus on exploring different aspects of the topic that are present in the provided chunks. Avoid questions that require external knowledge.

Suggested questions:
1.
2.
3.
"""
suggested_questions_prompt = PromptTemplate(
    template=suggested_questions_template,
    input_variables=["context", "chat_history", "relevant_chunks", "question"]
)

# Update the function to generate suggested questions
async def generate_suggested_questions(chain, question: str, context: str) -> List[str]:
    # Retrieve relevant chunks from the vector store
    relevant_docs = chain.retriever.get_relevant_documents(question)
    relevant_chunks = "\n".join([f"Chunk {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(relevant_docs[:3])])
    
    # Get chat history
    chat_history = chain.memory.chat_memory.messages
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-5:]])  # Last 5 messages
    
    suggested_questions_chain = LLMChain(llm=llm, prompt=suggested_questions_prompt)
    result = await suggested_questions_chain.arun(
        context=context,
        chat_history=chat_history_str,
        relevant_chunks=relevant_chunks,
        question=question
    )
    suggested_questions = [q.strip() for q in result.split('\n') if q.strip() and q.strip()[0].isdigit()]
    
    # Verify that suggested questions can be answered by the vector store
    verified_questions = []
    for q in suggested_questions:
        # Use the chain to check if the question can be answered
        result = await chain.acall(q)
        if result['answer'] and not result['answer'].lower().startswith("i'm sorry") and not result['answer'].lower().startswith("i apologize"):
            verified_questions.append(q)
        
        if len(verified_questions) == 3:
            break
    
    return verified_questions

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
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,
                                                   strip_headers=False,
                                                #    return_each_line=True
                                                )
    #########################

        # Char-level splits
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

    # Split
    # splits = text_splitter.split_documents(md_header_splits)

    ####################
    

    texts = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        splits = text_splitter.split_documents(splits)
        for split in splits:
            texts.append(Document(page_content=split.page_content, metadata={**doc.metadata, **split.metadata}))

    # Ensure each text has metadata for source and page
    for text in texts:
        text.metadata["source"] = f"{text.metadata.get('source', 'PDF')}"
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
        weights=[0.2, 0.8]
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

# New prompt template for query complexity detection
query_complexity_template = """
Analyze the following user query and determine if it's a simple or complex question.
A simple question typically asks for straightforward information that can be answered directly.
A complex question may require multiple steps, comparisons, or in-depth analysis to answer fully.

User query: {question}

Is this a simple or complex question? Respond with either "simple" or "complex".
"""
query_complexity_prompt = PromptTemplate(
    template=query_complexity_template,
    input_variables=["question"]
)

# Function to determine query complexity
async def determine_query_complexity(question: str) -> str:
    complexity_chain = LLMChain(llm=llm, prompt=query_complexity_prompt)
    result = await complexity_chain.arun(question=question)
    return result.strip().lower()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    # Step 1: Determine query complexity
    msg = cl.Message(content="Analyzing query complexity...")
    await msg.send()
    
    query_complexity = await determine_query_complexity(message.content)
    msg.content = f"Query determined to be: {query_complexity}"
    await msg.update()

    if query_complexity == "simple":
        # For simple queries, perform direct hybrid search
        msg.content = "Performing hybrid search for simple query..."
        await msg.update()
        
        result = await answer_question(chain, message.content)
        final_answer = result['answer']
        source_documents = result.get('source_documents', [])
    else:
        # For complex queries, use the existing decomposition and sub-question approach
        msg.content = "Decomposing the complex query into sub-questions..."
        await msg.update()
        
        sub_questions = await decompose_query(message.content)
        msg.content = f"Sub-questions: {', '.join(sub_questions)}"
        await msg.update()

        msg.content = "Performing hybrid search on sub-questions..."
        await msg.update()

        tasks = [answer_question(chain, q) for q in sub_questions]
        sub_answers = await asyncio.gather(*tasks)

        msg.content = "Combining sub-answers..."
        await msg.update()

        combined_answer = "\n\n".join([f"Q: {q}\nA: {a['answer']}" for q, a in zip(sub_questions, sub_answers)])

        msg.content = "Generating the final answer..."
        await msg.update()

        final_answer_prompt = PromptTemplate(
            template="Based on the following sub-questions and answers, provide the direct answer to the main question: {main_question}\n\nSub-questions and answers:\n{combined_answer}\n\nUse Pointer to presnt you answer where you feel necesary or in clean way. Dont provide unnecessay information. Final answer:",
            input_variables=["main_question", "combined_answer"]
        )
        final_answer_chain = LLMChain(llm=llm, prompt=final_answer_prompt)
        final_answer = await final_answer_chain.arun(main_question=message.content, combined_answer=combined_answer)

        source_documents = []
        for sub_answer in sub_answers:
            source_documents.extend(sub_answer.get("source_documents", []))

    # Handle source documents for citation
    unique_sources = {}
    for doc in source_documents:
        source_key = (doc.metadata['source'], doc.metadata['page'])
        if source_key not in unique_sources:
            unique_sources[source_key] = doc

    # Prepare source citations and elements
    source_citations = []
    text_elements = []
    for idx, (source_key, doc) in enumerate(unique_sources.items()):
        source_id = f"source_{idx+1}"
        source_citation = f"[{idx + 1}] {source_key[0]} (p. {source_key[1]})"
        source_citations.append(source_citation)
        
        # Create a Text element for each source
        source_content = f"Source: {source_key[0]}, Page: {source_key[1]}\n\n{doc.page_content}"
        text_elements.append(cl.Text(content=source_content, name=source_id))

    # Add source citations to the final answer
    if source_citations:
        final_answer += f"\n\nSources: {', '.join(source_citations)}"
    else:
        final_answer += "\nNo sources found."

    # Send the final answer
    await cl.Message(content=final_answer).send()

    # Send source texts as separate, collapsible messages
    if text_elements:
        for idx, element in enumerate(text_elements):
            await cl.Message(
                content=f"Source {idx + 1}: {source_citations[idx]}",
                elements=[element]
            ).send()
    # Generate suggested questions
    context = final_answer
    suggested_questions = await generate_suggested_questions(chain, message.content, context)

    elements = [
        cl.Text(content=question, title=f"Suggestion {i+1}")
        for i, question in enumerate(suggested_questions)
    ]

    await cl.Message(
        content="Here are some suggested follow-up questions:",
        elements=elements
    ).send()