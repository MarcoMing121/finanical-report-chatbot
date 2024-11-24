from typing import List, Dict
import os
import requests
import json

class DatabaseConnector:
    def __init__(self, base_url: str = "http://localhost:8080/api/documents"):
        self.base_url = base_url
    
    def index_documents(self, file_path: str = 'output/documents_5.json') -> str:
        """
        Index documents into database
        Args:
            file_path: Path to JSON file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Send to server
        files = {'file': open(file_path, 'rb')}
        response = requests.post(f"{self.base_url}/index", files=files)
        
        return response
    
    def query_documents(self, query_data: Dict) -> List[Dict]:
        """
        Query relevant documents
        Args:
            query_data: Dictionary containing query content and embedding vector
        """
        query_data = [query_data]
        # Save query as temporary JSON file
        with open('temp_query.json', 'w') as f:
            json.dump(query_data, f)
        
        # Send query request
        files = {'file': open('temp_query.json', 'rb')}
        response = requests.post(f"{self.base_url}/query", files=files)
        
        # Clean up temp file
        os.remove('temp_query.json')
        
        return response
    
def main(mode='query', file_path=None, query_data=None):
    db_connector = DatabaseConnector()

    if mode == 'index':
        if not file_path:
            print("Error: Index mode requires file_path")
            return None
        # Index documents
        print(f"Indexing file: {file_path}")
        result = db_connector.index_documents(file_path) 
        print(f"Index result: {result}")
        return result
    
    elif mode == 'query' and query_data:
        response = db_connector.query_documents(query_data)
        return response
    
    else:
        print("Error: Invalid mode or missing parameters")
        print("Usage:")
        print("  Index mode: main('index', file_path='path/to/file.json')")
        print("  Query mode: main('query', query_data=query_dict)")
        return None