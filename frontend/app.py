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