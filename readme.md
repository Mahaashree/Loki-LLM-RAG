# Loki - Document Q&A System

A streamlined document question-answering system using RAG (Retrieval-Augmented Generation) with LangChain and ChromaDB.

![image](https://github.com/user-attachments/assets/73d6576a-9027-404d-bddc-20f0db5a6641)
![image](https://github.com/user-attachments/assets/6ff4ba83-58ba-48b7-add1-20507d022f6e)


## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Mahaashree/Loki-LLM-RAG.git
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

| Category | File Types |
|----------|------------|
| Documents | PDF, DOCX, DOC |
| Spreadsheets | CSV, XLSX, XLS |
| Presentations | PPTX, PPT |
| Text | TXT, MD, HTML |
| Email | EML |

  ## 🏗️ Architecture

```
                                     ┌─────────────────┐
                                     │                 │
                                     │  Together AI    │
                                     │     (LLM)       │
                                     │                 │
                                     └────────┬────────┘
                                             │
┌─────────────┐    ┌──────────────┐    ┌────┴─────┐    ┌──────────────┐
│  Document   │    │   Document    │    │          │    │              │
│  Upload     ├───►│  Processing   ├───►│   RAG    ├───►│    Chat      │
│             │    │   Pipeline    │    │          │    │  Interface   │
└─────────────┘    └──────┬───────┘    └────┬─────┘    └──────────────┘
                          │                   │
                   ┌──────┴───────┐    ┌─────┴────────┐
                   │              │    │              │
                   │  ChromaDB    │◄───┤  Embeddings  │
                   │   Vector     │    │   Model      │
                   │    Store     │    │              │
                   └──────────────┘    └──────────────┘
```
![image](https://github.com/user-attachments/assets/f22dcae4-a683-4a65-9539-485fbfcffe0f)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License
