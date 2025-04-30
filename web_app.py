import streamlit as st
import os
import glob
import time
import re
from improved_rag_pipeline import (
    process_documents,
    setup_vector_db,
    create_qa_chain,
    summarize_document,
    find_relevant_documents
)

# Configure the page
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_added" not in st.session_state:
    st.session_state.documents_added = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar - Document Upload and Management
with st.sidebar:
    st.title("📚 Document Management")
    
    # Document Upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to add to the knowledge base",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv", "docx", "pptx", "xlsx", "md", "html"]
    )
    
    if uploaded_files and st.button("Process Documents"):
        st.session_state.processing = True
        
        # Save uploaded files temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Process and add to vector database
        with st.spinner("Processing documents..."):
            docs = process_documents(file_paths)
            if docs:
                vectordb = setup_vector_db(docs)
                st.session_state.documents_added = True
                st.success(f"Added {len(docs)} document chunks to the knowledge base")
            else:
                st.error("No documents were processed. Check file formats.")
        
        st.session_state.processing = False
    
    # Document folder path
    st.subheader("Add Local Documents")
    folder_path = st.text_input("Enter folder path to process")
    recursive = st.checkbox("Process recursively")
    
    if folder_path and st.button("Process Folder"):
        st.session_state.processing = True
        
        if os.path.exists(folder_path):
            with st.spinner("Processing documents from folder..."):
                if recursive:
                    file_paths = []
                    for root, _, filenames in os.walk(folder_path):
                        for filename in filenames:
                            file_paths.append(os.path.join(root, filename))
                else:
                    file_paths = glob.glob(os.path.join(folder_path, "*"))
                
                docs = process_documents(file_paths)
                if docs:
                    vectordb = setup_vector_db(docs)
                    st.session_state.documents_added = True
                    st.success(f"Added {len(docs)} document chunks from {len(file_paths)} files")
                else:
                    st.error("No documents were processed. Check folder contents.")
        else:
            st.error("Folder path does not exist")
        
        st.session_state.processing = False
    
    # Document browser
    if st.session_state.documents_added or os.path.exists("docs/chroma"):
        st.subheader("Browse Documents")
        vectordb = setup_vector_db()
        all_docs = vectordb.get()
        
        sources = set()
        for metadata in all_docs['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
        
        st.write(f"Knowledge base contains {len(sources)} documents:")
        for source in sorted(sources):
            st.write(f"- {os.path.basename(source)}")

# Main area
st.title("🔍 Document Assistant")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Chat", "Summarize", "Find"])

# Tab 1: Chat interface
# Functions for clearing chat history
def clear_chat():
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
with tab1:
    st.header("Chat with your Documents")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            clear_chat()
    
    # Check if documents are added
    if not st.session_state.documents_added and not os.path.exists("docs/chroma"):
        st.warning("Please add documents using the sidebar before chatting.")
    elif st.session_state.processing:
        st.info("Documents are being processed. Please wait...")
    else:
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
        
        # User input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user question to chat history
            with st.chat_message("user"):
                st.write(user_question)
            
            # Get answer from RAG pipeline
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                sources_placeholder = st.empty()
                
                with st.spinner("Thinking..."):
                    # Create QA chain
                    vectordb = setup_vector_db()
                    qa_chain = create_qa_chain(vectordb)
                    
                    # Get answer
                    start_time = time.time()
                    result = qa_chain.invoke({"question": user_question})
                    end_time = time.time()
                    
                    # Display answer
                    answer_placeholder.write(result['answer'])
                    
                    # Display sources
                    if 'source_documents' in result:
                        sources = set()
                        for doc in result['source_documents']:
                            if 'source' in doc.metadata:
                                sources.add(doc.metadata['source'])
                        
                        if sources:
                            sources_text = "Sources:\n" + "\n".join([f"- {os.path.basename(source)}" for source in sources])
                            sources_placeholder.info(sources_text)
                    
                    # Display time taken
                    st.caption(f"Time taken: {end_time - start_time:.2f} seconds")
            
            # Add to chat history
            st.session_state.chat_history.append((user_question, result['answer']))