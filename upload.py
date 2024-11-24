from connector import DatabaseConnector, main as connector_main
import json
import os
from pathlib import Path

def test_upload(start_idx=0, end_idx=2):
    """
    Test uploading JSON files within specified range
    Args:
        start_idx: Starting file index (inclusive)
        end_idx: Ending file index (inclusive) 
    """
    try:
        # Iterate through files in specified range
        for i in range(start_idx, end_idx + 1):
            source_file = Path(f"output/documents_{i}.json")
            if source_file.exists():
                # Read and check record count
                with open(source_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    record_count = len(data)
                    print(f"\ndocuments_{i}.json contains {record_count} records")
                
                print(f"Uploading documents_{i}.json...")
                response = connector_main('index', file_path=str(source_file))
                
                if response and response.status_code == 200:
                    print(f"documents_{i}.json uploaded successfully")
                else:
                    status = response.status_code if response else 'No response'
                    print(f"documents_{i}.json upload failed: {status}")
            else:
                print(f"\nFile documents_{i}.json does not exist")

    except Exception as e:
        print(f"Error processing file: {str(e)}")

def check_file_content(file_idx):
    """
    Check content of specified file
    """
    file_path = Path(f"output/documents_{file_idx}.json")
    if not file_path.exists():
        print(f"File documents_{file_idx}.json does not exist")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"\nFile documents_{file_idx}.json:")
            print(f"Record count: {len(data)}")
            if len(data) > 0:
                print("First record example:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    # Test uploading specific files
    test_upload(0, 13)  # Upload documents_0.json to documents_2.json
    
