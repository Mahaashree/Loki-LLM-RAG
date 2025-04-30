import os
import glob
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredEmailLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain,
    RetrievalQA,
    AnalyzeDocumentChain,
    LLMChain
)
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.base import LLM

# API client
from together import Together
from dotenv import load_dotenv

# Load environment variable
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)

# Vector DB configuration
PERSIST_DIRECTORY = 'docs/chroma/'
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  
LLM_MODEL = "meta-llama/Llama-Vision-Free"  

# Custom LLM class using Together API
class TogetherLLM(LLM):
    model: str
    temperature: float = 0.1
    max_tokens: int = 1024
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Together API: {e}")
            return "I encountered an error processing your request."

    @property
    def _llm_type(self) -> str:
        return "together"


def get_loader_for_file(file_path: str):
    """Return the appropriate loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        loaders = {
            '.pdf': PyPDFLoader(file_path),
            '.txt': TextLoader(file_path),
            '.csv': CSVLoader(file_path),
            '.docx': UnstructuredWordDocumentLoader(file_path),
            '.doc': UnstructuredWordDocumentLoader(file_path),
            '.pptx': UnstructuredPowerPointLoader(file_path),
            '.ppt': UnstructuredPowerPointLoader(file_path),
            '.xlsx': UnstructuredExcelLoader(file_path),
            '.xls': UnstructuredExcelLoader(file_path),
            '.md': UnstructuredMarkdownLoader(file_path),
            '.html': UnstructuredHTMLLoader(file_path),
            '.htm': UnstructuredHTMLLoader(file_path),
            '.eml': UnstructuredEmailLoader(file_path),
        }
        
        return loaders.get(ext)
    except Exception as e:
        print(f"Error creating loader for {file_path}: {e}")
        return None


def process_documents(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Process documents from file paths, add metadata, and split into chunks."""
    docs = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            continue
            
        loader = get_loader_for_file(file_path)
        if not loader:
            print(f"Unsupported file format: {file_path}")
            continue
        
        try:
            file_docs = loader.load()
            
            # Add metadata
            for doc in file_docs:
                doc.metadata["source"] = file_path
                doc.metadata["filename"] = os.path.basename(file_path)
                doc.metadata["filetype"] = os.path.splitext(file_path)[1]
                doc.metadata["created_at"] = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                doc.metadata["modified_at"] = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                doc.metadata["file_size"] = os.path.getsize(file_path)
            
            docs.extend(file_docs)
            print(f"Loaded {len(file_docs)} sections from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not docs:
        return []
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(splits)} chunks")
    
    return splits


def setup_vector_db(documents: Optional[List[Document]] = None) -> Chroma:
    """Set up or load the vector database."""
    # Initialize embeddings with proper model configuration
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'device': 'cpu', 'batch_size': 32}
    )
    
    # If persist directory exists and no new documents, load existing DB
    if os.path.exists(PERSIST_DIRECTORY) and documents is None:
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        print(f"Loaded vector database with {vectordb._collection.count()} chunks")
        return vectordb
    
    # Create directory if it doesn't exist
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    if documents:
        # Create or update vector database with new documents
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"Vector database updated with {vectordb._collection.count()} chunks")
    else:
        # Create empty vector database
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        print(f"Created new empty vector database")
    
    return vectordb


def create_qa_chain(vectordb: Chroma, chain_type: str = "stuff") -> Any:
    """Create a QA chain based on the specified chain type."""
    # Initialize LLM with streaming capability
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = TogetherLLM(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=1024
    )
    
    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create retriever with MMR search for better diversity
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )
    
    # Define comprehensive prompt
    template = """You are a document assistant that helps users find and understand information in their files.

CONTEXT INFORMATION:
{context}

CHAT HISTORY:
{chat_history}

USER QUERY: {question}

INSTRUCTIONS:
1. Answer ONLY using information from the context.
2. If the answer isn't in the context, say "I don't have enough information about that in these documents."
3. Include the source file name for each piece of information you provide.
4. Respond in a clear, concise manner.
5. If the query is about finding files, list relevant files with their paths.

ANSWER:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Create appropriate chain based on chain_type
    if chain_type == "stuff":
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
    else:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
    
    return qa_chain


def summarize_document(file_path: str) -> str:
    """Generate a summary for a specific document."""
    loader = get_loader_for_file(file_path)
    if not loader:
        return f"Unsupported file format: {file_path}"
    
    try:
        docs = loader.load()
        
        llm = TogetherLLM(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        
        # Create a summarization prompt
        summary_prompt = PromptTemplate.from_template(
            "You are a document summarization expert. Provide a comprehensive summary of the following document:\n\n"
            "{text}\n\n"
            "Your summary should include:\n"
            "1. Main topics and key points\n"
            "2. Important findings or conclusions\n"
            "3. Any actionable information\n\n"
            "Summary:"
        )
        
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
        
        # Combine all document sections
        full_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate summary
        summary = document_chain.run(full_text)
        return summary
    
    except Exception as e:
        return f"Error summarizing document: {e}"


def find_relevant_documents(query: str, vectordb: Chroma, k: int = 5) -> List[Document]:
    """Find documents relevant to a query using semantic search."""
    docs = vectordb.similarity_search(query, k=k)
    return docs


def interactive_mode():
    """Start an interactive session with the RAG system."""
    print("Welcome to the Document Assistant!")
    print("Type 'exit' to quit, 'help' for commands")
    
    # Load vector database
    vectordb = setup_vector_db()
    
    # Create QA chain
    qa_chain = create_qa_chain(vectordb)
    
    while True:
        query = input("\nQuestion: ")
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'help':
            print("\nCommands:")
            print("  help - Show this help")
            print("  exit - Exit the program")
            print("  add <path> - Add a document or directory")
            print("  list - List indexed documents")
            print("  summarize <path> - Summarize a document")
            print("  find <keyword> - Find documents mentioning keyword")
            continue
        elif query.lower().startswith('add '):
            path = query[4:].strip()
            if os.path.exists(path):
                if os.path.isfile(path):
                    docs = process_documents([path])
                else:
                    file_paths = glob.glob(os.path.join(path, "*"))
                    docs = process_documents(file_paths)
                
                if docs:
                    vectordb = setup_vector_db(docs)
                    print(f"Added document(s) from {path}")
                else:
                    print("No documents were added")
            else:
                print(f"Path does not exist: {path}")
            continue
        elif query.lower() == 'list':
            # List all documents in the vector database
            all_docs = vectordb.get()
            sources = set()
            for metadata in all_docs['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            
            print(f"Knowledge base contains {len(sources)} documents:")
            for source in sorted(sources):
                print(f"- {source}")
            continue
        elif query.lower().startswith('summarize '):
            path = query[10:].strip()
            if os.path.exists(path) and os.path.isfile(path):
                print(f"Summarizing {path}...")
                summary = summarize_document(path)
                print(f"\nSummary of {os.path.basename(path)}:")
                print(summary)
            else:
                print(f"File does not exist: {path}")
            continue
        elif query.lower().startswith('find '):
            keyword = query[5:].strip()
            docs = find_relevant_documents(keyword, vectordb)
            
            print(f"\nDocuments relevant to '{keyword}':")
            sources = set()
            for doc in docs:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            for source in sources:
                print(f"- {source}")
            
            if not sources:
                print("No matching documents found.")
            continue
        
        # Regular query
        try:
            start_time = time.time()
            result = qa_chain({"question": query})
            end_time = time.time()
            
            print(f"\nAnswer: {result['answer']}")
            
            if 'source_documents' in result:
                print("\nSources:")
                sources = set()
                for doc in result['source_documents']:
                    if 'source' in doc.metadata:
                        sources.add(doc.metadata['source'])
                for source in sources:
                    print(f"- {source}")
            
            print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    interactive_mode()