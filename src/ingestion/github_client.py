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
        '.md', '.rst', '.txt', '.json', '.yaml', '.yml', '.html', '.css', '.scss', '.vue', '.rb', '.go', '.php', '.swift', '.kt', '.rs', '.sh', '.bash', '.sql', '.xml', '.csharp', '.cs', '.dart', '.scala', '.pl', '.lua', '.groovy', '.perl', '.clj', '.clojure', '.elixir', '.exs', '.erl', '.fs', '.fsx', '.tsx', '.jsx', '.vue', '.svelte', '.asm', '.asmx', '.m', '.mm', '.swift', '.hlsl', '.glsl', '.shader'
    ]
    
    return any(filename.endswith(ext) for ext in relevant_extensions)