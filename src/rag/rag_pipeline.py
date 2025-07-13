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