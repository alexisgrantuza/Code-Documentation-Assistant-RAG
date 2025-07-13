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