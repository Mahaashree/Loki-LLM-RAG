# Loki - Document Q&A System

A streamlined document question-answering system using RAG (Retrieval-Augmented Generation) with LangChain and ChromaDB.

## Quick Start

1. **Clone the repository**
```bash
git clone [https://github.com/Mahaashree/Loki-LLM-RAG.git](https://github.com/Mahaashree/Loki-LLM-RAG.git)
cd Loki-LLM-RAG
```

2. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup env variables**
```bash
TOGETHER_API_KEY=your_api_key_here
```

4. **Run the app**
```bash
streamlit run web_app.py
```
or
```bash
streamlit run web_app.py --server.fileWatcherType none
```

## Features

- Document upload and management
- Document browsing
- Chat interface with document retrieval
- Document summarization
- Document finding

## Supported File Types

- PDF documents
- Text files
- Word documents
- PowerPoint presentations
- Excel spreadsheets
- HTML files
- Markdown files
- CSV files
- Email files

## License

MIT License
