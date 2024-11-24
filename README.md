# Financial Report Chatbot

A RAG (Retrieval-Augmented Generation) system for querying and analyzing financial reports, built with LangChain and distributed vector database.

## Features
- Process and analyze Hong Kong listed companies' financial reports
- Distributed vector database for efficient document retrieval
- LDA topic modeling for enhanced document understanding
- Interactive chat interface with Chainlit

## Prerequisites
- Python 3.10+
- CUDA-capable GPU
- Ubuntu 22.04 (recommended)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/MarcoMing121/finanical-report-chatbot.git
cd finanical-report-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required data files and place them in the input directory:
```
input/
  ├── lda_topic_distribution/
  └── txt_data/
```

5. Run the application:
```bash
chainlit run chat-with-upload.py
```

## Usage
1. Open browser and navigate to `http://localhost:8000`
2. Enter your query about financial reports
3. System will retrieve relevant information and generate response

## Note
Large data files are not included in this repository. Please contact the maintainer for access to the required data files.