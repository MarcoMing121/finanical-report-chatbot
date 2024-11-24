from typing import List
import chainlit as cl
from connector import DatabaseConnector, main as connector_main
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA
import json
from pathlib import Path
import pandas as pd
import os
import urllib.parse
import uuid
from datetime import datetime
import time

def load_keyword_data(path):
    """Load keyword data from all parquet files"""
    keyword_dict = {}
    for parquet_file in Path(path).glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        df['file_name'] = df['file_name'].apply(lambda x: urllib.parse.unquote(x))
        for _, row in df.iterrows():
            keyword_dict[row['file_name']] = ', '.join(row['all_words'])
    return keyword_dict

def load_model():
    llm = Ollama(
        model="mixtral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

# Define document content description at the beginning of file
document_content_description = """
This is a collection of financial reports from Hong Kong listed companies. Each document contains:
1. Company name and stock code
2. Report type (Annual/Interim Report) and year
3. Financial performance and business conditions
4. Business operations details
5. Financial statement data

Document metadata includes:
- filename: File name (contains company name, stock code, report type and year)
- publish_date: Report publication date (Unix timestamp format)
- chunk_id: Document chunk sequence number
- keywords: Keywords extracted using LDA topic modeling
"""

metadata_field_info = [
    AttributeInfo(
        name="filename",
        description="Filename format: CompanyName_StockCode_ReportTypeYear, e.g., 'TencentHoldings_00700HK_Annual Report 2022'",
        type="string",
    ),
    AttributeInfo(
        name="publish_date",
        description="Report publication timestamp",
        type="number",
    ),
    AttributeInfo(
        name="chunk_id",
        description="Sequence number indicating the position of this chunk in the original document",
        type="number",
    ),
    AttributeInfo(
        name="keywords", 
        description="Keywords extracted from the document using LDA topic modeling", 
        type="string"
    ),
]

@cl.on_chat_start
async def on_chat_start():
    try:
        # 1. Load LDA topic distribution data
        keyword_data = load_keyword_data("input/lda_topic_distribution")
        cl.user_session.set("keyword_data", keyword_data)
        
        # 2. Initialize models
        llm = load_model()
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/quora-distilbert-base",
            model_kwargs={"device": "cuda"}
        )
        cl.user_session.set("llm", llm)
        cl.user_session.set("embedding_model", embedding_model)
        
        await cl.Message(content="System is ready. Please enter your query.").send()
            
    except Exception as e:
        await cl.Message(content=f"System initialization error: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    start_time = time.time()
    processing_msg = cl.Message(content="Processing your query...")
    await processing_msg.send()
    
    try:
        # 1. Generate embeddings
        embedding_time_start = time.time()
        embedding_model = cl.user_session.get("embedding_model")
        query = [message.content]
        query_embeddings = embedding_model.embed_documents(query)
        query_data = {
            "id": str(uuid.uuid4()),
            "content": query[0],
            "metadata": {},
            "embedding": query_embeddings[0]
        }
        embedding_time = time.time() - embedding_time_start

        # 2. Server communication timing
        server_time_start = time.time()
        send_time_start = time.time()
        response = connector_main('query', query_data=query_data)
        send_time = time.time() - send_time_start
        
        if not response or response.status_code != 200:
            await cl.Message(content="Server query failed").send()
            return
            
        receive_time = time.time() - (send_time_start + send_time)
        total_server_time = time.time() - server_time_start

        # 3. Process returned documents, add LDA topic keywords
        keyword_data = cl.user_session.get("keyword_data")
        docs = []
        results = json.loads(response.text)
        
        for result in results:
            metadata = result.get('metadata', {})
            if metadata.get('filename') in keyword_data:
                metadata['keywords'] = keyword_data[metadata['filename']]
            else:
                metadata['keywords'] = ""
            doc = Document(
                page_content=result.get('content', ''),
                metadata=metadata
            )
            docs.append(doc)

        if not docs:
            await cl.Message(content="No relevant documents found").send()
            return

        # 4. Create local vector store for current session
        vectorstore = Chroma.from_documents(docs, embedding_model)
        
        # 5. Set up retriever
        retriever = SelfQueryRetriever.from_llm(
            cl.user_session.get("llm"),
            vectorstore,
            document_content_description,
            metadata_field_info,
            verbose=True
        )

        # 6. Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            cl.user_session.get("llm"),
            retriever=retriever,
            return_source_documents=True,
        )

        # 7. Execute QA (with streaming)
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["Here's my answer:\n"]
        )
        
        msg = cl.Message(content="")
        await msg.send()
        
        qa_time_start = time.time()
        res = await qa_chain.ainvoke(
            message.content, 
            callbacks=[cb]
        )
        qa_time = time.time() - qa_time_start
        
        # 8. Process results
        answer = res["result"]
        source_documents = res["source_documents"]

        # Calculate total time
        total_time = time.time() - start_time
        
        # 10. Send final response (with timing stats)
        time_stats = f"""
Processing Time Statistics:
- Embedding Generation: {embedding_time:.2f}s
- Server Communication:
  • Query Send Time: {send_time:.2f}s
  • Response Receive Time: {receive_time:.2f}s
  • Total Server Time: {total_server_time:.2f}s
- QA Generation: {qa_time:.2f}s
- Total Processing Time: {total_time:.2f}s
"""
        
        # Add request/response details for debugging
        request_details = f"""
Request/Response Details:
- Request Size: {len(str(query_data))} bytes
- Response Size: {len(response.text)} bytes
- Status Code: {response.status_code}
- Server Response Time: {response.elapsed.total_seconds():.2f}s
"""
        
        await cl.Message(
            content=f"{answer}\n\n{time_stats}\n{request_details}"
        ).send()

    except Exception as e:
        await cl.Message(content=f"Error processing query: {str(e)}").send()
    
    finally:
        await processing_msg.remove()

@cl.on_chat_end
async def end():
    # Clean up temporary files
    if os.path.exists("temp_query.json"):
        os.remove("temp_query.json")