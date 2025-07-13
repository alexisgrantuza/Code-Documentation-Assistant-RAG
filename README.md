# Code Documentation Assistant - Complete Setup Guide

## Prerequisites

- Python 3.8+
- Git
- OpenAI API key
- GitHub Personal Access Token (optional, for private repos)

## 1. Project Structure Setup

First, create the project directory structure:

```bash
mkdir code-doc-assistant
cd code-doc-assistant

# Create directory structure
mkdir -p src/{ingestion,vectorstore,rag,api}
mkdir -p config
mkdir -p data
mkdir -p frontend
touch requirements.txt
touch .env
touch README.md
```

## 2. Dependencies Installation

Create `requirements.txt`:

```txt
# Core AI/ML libraries
langchain==0.1.0
langchain-openai==0.0.5
langchain-community==0.0.13
openai==1.10.0

# Vector database
faiss-cpu==1.7.4
chromadb==0.4.22

# Web framework
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.0

# GitHub integration
PyGithub==2.1.1
requests==2.31.0

# Text processing
tiktoken==0.5.2
python-dotenv==1.0.0

# Frontend
streamlit==1.31.0

# Utilities
python-multipart==0.0.6
aiofiles==23.2.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Environment Configuration

Create `.env` file:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# GitHub API (optional - for private repos)
GITHUB_TOKEN=your_github_token_here

# Vector Database
VECTOR_DB_PATH=./data/vectordb

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 4. Core Components Implementation

### 4.1 Configuration Module

Create `config/settings.py`:

```python
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/vectordb")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

settings = Settings()
```

### 4.2 GitHub Integration

Create `src/ingestion/github_client.py`:

```python
import os
import requests
from typing import List, Dict, Optional
from github import Github, Repository
from config.settings import settings

class GitHubClient:
    def __init__(self):
        self.github = Github(settings.github_token) if settings.github_token else Github()

    def get_repo_content(self, repo_url: str) -> Dict:
        """Extract repository content from GitHub URL"""
        # Parse repo URL
        repo_name = repo_url.replace("https://github.com/", "").replace(".git", "")

        try:
            repo = self.github.get_repo(repo_name)
            content = self._extract_repo_files(repo)
            return {
                "repo_name": repo_name,
                "description": repo.description,
                "files": content
            }
        except Exception as e:
            print(f"Error fetching repo: {e}")
            return None

    def _extract_repo_files(self, repo: Repository) -> List[Dict]:
        """Extract files from repository"""
        files = []

        # Get all files recursively
        contents = repo.get_contents("")

        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                # Filter relevant files
                if self._is_relevant_file(file_content.name):
                    try:
                        files.append({
                            "path": file_content.path,
                            "name": file_content.name,
                            "content": file_content.decoded_content.decode('utf-8'),
                            "size": file_content.size
                        })
                    except:
                        continue

        return files

    def _is_relevant_file(self, filename: str) -> bool:
        """Check if file is relevant for documentation"""
        relevant_extensions = [
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
            '.md', '.rst', '.txt', '.json', '.yaml', '.yml'
        ]

        return any(filename.endswith(ext) for ext in relevant_extensions)
```

### 4.3 Document Processing

Create `src/ingestion/document_processor.py`:

```python
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import settings

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_repo_files(self, repo_data: Dict) -> List[Document]:
        """Process repository files into documents"""
        documents = []

        for file_info in repo_data["files"]:
            # Create chunks for each file
            chunks = self.text_splitter.split_text(file_info["content"])

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "repo_name": repo_data["repo_name"],
                        "file_path": file_info["path"],
                        "file_name": file_info["name"],
                        "chunk_index": i,
                        "file_size": file_info["size"]
                    }
                )
                documents.append(doc)

        return documents

    def extract_code_context(self, content: str, file_path: str) -> Dict:
        """Extract additional context from code files"""
        context = {
            "functions": [],
            "classes": [],
            "imports": []
        }

        lines = content.split('\n')

        for line in lines:
            line = line.strip()

            # Extract function definitions
            if line.startswith('def '):
                context["functions"].append(line)

            # Extract class definitions
            elif line.startswith('class '):
                context["classes"].append(line)

            # Extract imports
            elif line.startswith('import ') or line.startswith('from '):
                context["imports"].append(line)

        return context
```

### 4.4 Vector Store

Create `src/vectorstore/vector_db.py`:

```python
import os
import pickle
from typing import List, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from config.settings import settings

class VectorDB:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key
        )
        self.vectorstore = None
        self.db_path = settings.vector_db_path

    def create_vectorstore(self, documents: List[Document]) -> bool:
        """Create vector store from documents"""
        try:
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embeddings
            )
            return True
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False

    def save_vectorstore(self, repo_name: str):
        """Save vector store to disk"""
        if self.vectorstore:
            os.makedirs(self.db_path, exist_ok=True)
            save_path = os.path.join(self.db_path, f"{repo_name}.faiss")
            self.vectorstore.save_local(save_path)

    def load_vectorstore(self, repo_name: str) -> bool:
        """Load vector store from disk"""
        try:
            load_path = os.path.join(self.db_path, f"{repo_name}.faiss")
            if os.path.exists(load_path):
                self.vectorstore = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            return False
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        return []

    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store"""
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
```

### 4.5 RAG Pipeline

Create `src/rag/rag_pipeline.py`:

```python
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from config.settings import settings

class RAGPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=settings.openai_api_key,
            temperature=0.1
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful code documentation assistant.
            Your job is to help developers understand codebases by answering questions about code.

            Use the following context from the codebase to answer the question:
            {context}

            Guidelines:
            - Be precise and technical
            - Include relevant code snippets when helpful
            - Reference specific files when possible
            - If you're unsure, say so
            - Focus on practical, actionable information"""),
            ("human", "{question}")
        ])

    def generate_response(self, question: str, context_docs: List[Document]) -> str:
        """Generate response using RAG pipeline"""

        # Format context
        context = self._format_context(context_docs)

        # Create chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Generate response
        response = chain.invoke(question)
        return response

    def _format_context(self, docs: List[Document]) -> str:
        """Format documents into context string"""
        context_parts = []

        for doc in docs:
            file_path = doc.metadata.get("file_path", "unknown")
            content = doc.page_content

            context_parts.append(f"File: {file_path}\n{content}\n---")

        return "\n".join(context_parts)
```

### 4.6 FastAPI Backend

Create `src/api/main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

from src.ingestion.github_client import GitHubClient
from src.ingestion.document_processor import DocumentProcessor
from src.vectorstore.vector_db import VectorDB
from src.rag.rag_pipeline import RAGPipeline

app = FastAPI(title="Code Documentation Assistant")

# Initialize components
github_client = GitHubClient()
doc_processor = DocumentProcessor()
vector_db = VectorDB()
rag_pipeline = RAGPipeline()

class IndexRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    repo_name: str
    question: str

@app.post("/index")
async def index_repository(request: IndexRequest):
    """Index a GitHub repository"""
    try:
        # Extract repository data
        repo_data = github_client.get_repo_content(request.repo_url)
        if not repo_data:
            raise HTTPException(status_code=400, detail="Failed to fetch repository")

        # Process documents
        documents = doc_processor.process_repo_files(repo_data)

        # Create vector store
        if vector_db.create_vectorstore(documents):
            vector_db.save_vectorstore(repo_data["repo_name"])
            return {"message": f"Repository {repo_data['repo_name']} indexed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create vector store")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_repository(request: QueryRequest):
    """Query indexed repository"""
    try:
        # Load vector store
        if not vector_db.load_vectorstore(request.repo_name):
            raise HTTPException(status_code=404, detail="Repository not found")

        # Search for relevant documents
        relevant_docs = vector_db.similarity_search(request.question)

        # Generate response
        response = rag_pipeline.generate_response(request.question, relevant_docs)

        return {
            "question": request.question,
            "answer": response,
            "sources": [doc.metadata["file_path"] for doc in relevant_docs]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Code Documentation Assistant API"}

if __name__ == "__main__":
    import uvicorn
    from config.settings import settings

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
```

## 5. Simple Frontend (Streamlit)

Create `frontend/app.py`:

```python
import streamlit as st
import requests
import json

st.set_page_config(page_title="Code Documentation Assistant", layout="wide")

st.title("🤖 Code Documentation Assistant")
st.markdown("Ask questions about any GitHub repository!")

# API endpoint
API_BASE = "http://localhost:8000"

# Sidebar for repository management
st.sidebar.header("Repository Management")

repo_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")

if st.sidebar.button("Index Repository"):
    if repo_url:
        with st.spinner("Indexing repository..."):
            try:
                response = requests.post(f"{API_BASE}/index", json={"repo_url": repo_url})
                if response.status_code == 200:
                    st.sidebar.success("Repository indexed successfully!")
                else:
                    st.sidebar.error(f"Error: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

# Main chat interface
st.header("Ask Questions")

repo_name = st.text_input("Repository Name", placeholder="user/repo")
question = st.text_area("Your Question", placeholder="How does authentication work in this app?")

if st.button("Ask Question"):
    if repo_name and question:
        with st.spinner("Searching for answer..."):
            try:
                response = requests.post(f"{API_BASE}/query", json={
                    "repo_name": repo_name,
                    "question": question
                })

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Sources")
                    for source in result["sources"]:
                        st.code(source)

                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Instructions
st.markdown("---")
st.markdown("""
### How to use:
1. Enter a GitHub repository URL in the sidebar
2. Click "Index Repository" to process the code
3. Enter the repository name (e.g., "user/repo")
4. Ask questions about the code
""")
```

## 6. Running the Application

### Start the API server:

```bash
python src/api/main.py
```

### Start the Streamlit frontend:

```bash
streamlit run frontend/app.py
```

## 7. Usage Example

1. Index a repository: `https://github.com/fastapi/fastapi`
2. Ask questions like:
   - "How does FastAPI handle authentication?"
   - "What are the main components of this application?"
   - "Show me the database models"

## 8. Next Steps

- Add support for more file types
- Implement better code parsing
- Add memory for conversation history
- Improve chunking strategies for code
- Add unit tests
- Deploy to cloud platform

## Troubleshooting

- Make sure your OpenAI API key is set
- Check that the GitHub repository is public (or you have access)
- Ensure all dependencies are installed
- Check API logs for detailed error messages

I've created a comprehensive setup guide for your Code Documentation Assistant project. Here's a quick summary of what we've built:
Key Components:

GitHub Integration - Fetches repository content
Document Processing - Chunks code files intelligently
Vector Database - FAISS for semantic search
RAG Pipeline - LangChain + OpenAI for Q&A
FastAPI Backend - REST API for indexing and querying
Streamlit Frontend - Simple web interface

Quick Start:

Get your OpenAI API key from OpenAI
Set up the project structure as shown
Install dependencies: pip install -r requirements.txt
Create .env file with your API key
Run the API: python src/api/main.py
Run the frontend: streamlit run frontend/app.py

How It Works:

Index a GitHub repo (it downloads and processes all code files)
Ask questions like "How does authentication work?" or "What does the main function do?"
Get answers with relevant code snippets and file references

The project starts simple but gives you a solid foundation to build upon. You can enhance it by adding more sophisticated code parsing, better chunking strategies, or support for more programming languages.
Would you like me to help you with any specific part of the implementation, or do you have questions about getting started?
