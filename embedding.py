from typing import List, Dict
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import requests
from pathlib import Path
import datetime 
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

class DocumentProcessor:
    def __init__(
        self, 
        txt_folder: str, 
        parquet_folder: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ):
        self.txt_folder = txt_folder
        self.parquet_folder = parquet_folder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def _parse_metadata(self, first_line: str, filename) -> Dict:
        """Parse metadata from first line of file"""
        try:
            # Remove brackets and split
            content = first_line.strip().split("('")[1].split("',")[0], first_line.strip().split(",")[1].strip().split(")")[0]
            publish_date = datetime.datetime.fromtimestamp(float(content[1].strip()), tz=datetime.timezone.utc)
            return {
                "filename": filename,
                "publish_date": str(publish_date)
            }
        except Exception as e:
            print(f"Metadata parsing error: {e}")
            return {}
        
    def remove_files_in_folder(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    def process_documents(self) -> List[Document]:
        """Process all documents and return Document list (without embeddings)"""
        documents = []
        file_metadatas = []
        txt_files = list(Path(self.txt_folder).glob("*.txt"))
        
        for txt_file in tqdm(txt_files, desc="Processing documents"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                filename = str(txt_file).split('/')[-1]
                metadata = self._parse_metadata(first_line,filename)
                content = f.read()

                # if metadata.get('filename') in self.keyword_data:
                #     metadata['keywords'] = self.keyword_data[metadata['filename']]
                file_metadatas.append(metadata)
                
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            'chunk_id': i
                        }
                    )
                    documents.append(doc)
        
        return documents, file_metadatas
    
    def calculate_embeddings(self, documents: List[Document], batch_size: int = 64, output_file: str = 'documents.json') -> List[Document]:
        """Calculate document embeddings in batches and write to JSON file"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/quora-distilbert-base",
            model_kwargs={"device": "cuda"}
        )
        # Ensure output directory exists
        folder_name = "output"
        os.makedirs(folder_name, exist_ok=True)
        self.remove_files_in_folder(folder_name)
        
        # Collect all texts
        texts = [doc.page_content for doc in documents]
        processed_documents = []
        file_counter = 0
        total_processed = 0
        current_output = f"{folder_name}/{output_file.rsplit('.', 1)[0]}_{file_counter}.json"
        
        # Batch processing
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            
            try:
                batch_embeddings = embeddings.embed_documents(batch_texts)
                
                # Add embeddings to documents
                for doc, emb in zip(batch_docs, batch_embeddings):
                    doc.metadata["embedding"] = emb
                    processed_documents.append(doc)
                
                # Write to JSON file every 100 documents
                if len(processed_documents) >= 100*batch_size:
                    self.write_documents_to_json_batch(
                        processed_documents,
                        output_file=current_output
                    )
                    processed_documents.clear()
                    total_processed += 100*batch_size
                    
                    # Create new output file every 300 batches
                    if total_processed >= 300*batch_size:
                        file_counter += 1
                        current_output = f"{folder_name}/{output_file.rsplit('.', 1)[0]}_{file_counter}.json"
                        total_processed = 0
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue
        
        # Process remaining documents
        if processed_documents:
            self.write_documents_to_json_batch(
                processed_documents,
                output_file=current_output
            )
        processed_documents.clear()
                
        return 0
    def write_documents_to_json_batch(self, documents: List[Document], output_file: str = 'documents.json') -> None:
        """Write documents to JSON file in batches
        
        Args:
            documents: List of documents
            batch_size: Number of documents per batch
            output_file: Output JSON filename
        """
        # Create empty JSON array if file doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[]')
        print("Writing data to json")
        json_data = []
        
        # Process current batch of documents
        for doc in documents:
            json_data.append({
                "id": f"{doc.metadata['filename']}_{doc.metadata['chunk_id']}",
                "embedding": doc.metadata["embedding"],
                "metadata" : {k: v for k, v in doc.metadata.items() if k != "embedding"},
                "content": doc.page_content
            })
        
        # Open file in append mode and read last character
        with open(output_file, 'rb+') as f:
            # Move to second-to-last character
            f.seek(-1, 2)
            last_char = f.read().decode('utf-8')
            # Remove closing bracket if present
            if last_char == ']':
                f.seek(-1, 2)
                f.truncate()
            f.seek(-1, 2)
            last_char = f.read().decode('utf-8')
                
        # Open file in append mode
        with open(output_file, 'a', encoding='utf-8') as f:
            # Add comma if needed
            if last_char not in ['[', ',']:
                f.write(',')
            
            
            # Write current batch
            for i, item in enumerate(json_data):
                f.write(json.dumps(item, ensure_ascii=False))
                if i < len(json_data) - 1:
                    f.write(',')
                else:
                    f.write(']')
 
def main():
    # Usage example
    processor = DocumentProcessor(
        txt_folder="./input/txt_data_2",
        parquet_folder="./input/lda_topic_dsitribution"
    )
    
    # Process all documents
    # Step 1: Process document content
    documents, file_metadatas = processor.process_documents()
    print(f"Document processing complete, {len(documents)} documents total")
    with open('output/file_metadatas.json', 'w') as file:
        json.dump(file_metadatas, file)
    file_metadatas.clear()
    
    # Step 2: Calculate embeddings
    processor.calculate_embeddings(documents, batch_size=64)
    print(f"Embedding calculation complete")

if __name__ == "__main__":
    main()