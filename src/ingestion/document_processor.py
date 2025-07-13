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