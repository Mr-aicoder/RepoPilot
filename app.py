# main.py (or your Streamlit app's entry file)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st 
import os
import requests  
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from dotenv import load_dotenv  
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory # Import ConversationBufferMemory
import uuid # Import uuid for generating unique IDs
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import text splitter

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# IMPORTANT: Replace with your actual Groq API Key and GitHub Personal Access Token
# It's highly recommended to use environment variables for these.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # Needs 'repo' scope for private repos, 'public_repo' for public.

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it as an environment variable or in a .env file.")
if not GITHUB_TOKEN:
    st.warning("GITHUB_TOKEN not found. Some GitHub API calls (especially for private repos or higher rate limits) might fail.")

# --- Initialize all Streamlit session state variables at the very beginning ---
# This ensures they are always defined before being accessed.
if "messages" not in st.session_state: # Re-initialize chat history for single session
    st.session_state.messages = []
if "repo_url" not in st.session_state: # Re-initialize repo_url for single session
    st.session_state.repo_url = ""
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "vector_store" not in st.session_state: # Re-initialize vector_store for single session
    st.session_state.vector_store = None
if "vector_store_repo_url" not in st.session_state: # Re-initialize vector_store_repo_url for single session
    st.session_state.vector_store_repo_url = ""
if "debug_response" not in st.session_state: # New: To store response for debug/generation
    st.session_state.debug_response = ""


# --- GitHub API Interaction Functions ---

def parse_github_url(repo_url):
    """Parses a GitHub URL to extract owner and repo name."""
    parts = repo_url.strip('/').split('/')
    if len(parts) >= 2 and parts[-2] and parts[-1]:
        owner = parts[-2]
        repo_name = parts[-1].replace(".git", "") # Handle .git suffix
        return owner, repo_name
    return None, None

@tool
@st.cache_data(show_spinner=False) # Cache the result to avoid re-fetching on every rerun
def list_repo_files(repo_url: str) -> str:
    """
    Lists all files and directories in a GitHub repository recursively.
    Args:
        repo_url (str): The URL of the GitHub repository (e.g., 'https://github.com/owner/repo').
    Returns:
        str: A newline-separated string of file paths, or an error message.
    """
    owner, repo_name = parse_github_url(repo_url)
    if not owner or not repo_name:
        return "Invalid GitHub repository URL provided."

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        # Get the default branch
        repo_info_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        repo_response = requests.get(repo_info_url, headers=headers)
        repo_response.raise_for_status()
        default_branch = repo_response.json().get('default_branch', 'main')

        # Get the tree (recursive)
        tree_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{default_branch}?recursive=1"
        response = requests.get(tree_url, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        tree = response.json().get('tree', [])
        file_paths = [item['path'] for item in tree if item['type'] == 'blob']
        return "\n".join(file_paths)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: Repository '{owner}/{repo_name}' not found. Please check the URL."
        elif e.response.status_code == 403:
            return f"Error: Access forbidden to '{owner}/{repo_name}'. This might be a private repository or you've hit GitHub API rate limits. Please check your GITHUB_TOKEN and its permissions."
        return f"HTTP Error accessing GitHub repository: {e}. Status Code: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Network or connection error accessing GitHub repository: {e}. Please check your internet connection."
    except Exception as e:
        return f"An unexpected error occurred while listing files: {e}"

@tool
def get_file_content(repo_url: str, file_path: str) -> str:
    """
    Fetches the content of a specific file from a GitHub repository.
    Args:
        repo_url (str): The URL of the GitHub repository (e.g., 'https://github.com/owner/repo').
        file_path (str): The path to the file within the repository (e.g., 'src/main.py').
    Returns:
        str: The content of the file, or an error message.
    """
    owner, repo_name = parse_github_url(repo_url)
    if not owner or not repo_name:
        return "Invalid GitHub repository URL provided."

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        # Get the default branch
        repo_info_url = f"https://api.github.com/repos/{owner}/{repo_name}"
        repo_response = requests.get(repo_info_url, headers=headers)
        repo_response.raise_for_status()
        default_branch = repo_response.json().get('default_branch', 'main')

        file_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{default_branch}/{file_path}"
        response = requests.get(file_url, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: File '{file_path}' not found in repository '{owner}/{repo_name}'. Please ensure the file path is correct."
        elif e.response.status_code == 403:
            return f"Error: Access forbidden to file '{file_path}'. This might be due to GitHub API rate limits or insufficient permissions for your GITHUB_TOKEN (if it's a private repo)."
        return f"HTTP Error fetching file content: {e}. Status Code: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Network or connection error fetching file content: {e}. Please check your internet connection."
    except Exception as e:
        return f"An unexpected error occurred while getting file content: {e}"

@tool
def load_and_embed_repository(repo_url: str) -> str:
    """
    Loads all relevant file contents from a GitHub repository, splits them into chunks,
    and embeds them into a vector store for semantic search. This process can take time for large repositories.
    Args:
        repo_url (str): The URL of the GitHub repository.
    Returns:
        str: A confirmation message or an error if loading fails.
    """
    # Using st.session_state.vector_store directly for single session
    if st.session_state.vector_store_repo_url == repo_url and st.session_state.vector_store is not None:
        return "Repository content already loaded and embedded for this URL. You can now ask general questions."

    st.info(f"Loading and embedding repository content for {repo_url}. This may take a while...")
    file_list_result = list_repo_files.invoke(repo_url)
    if "Error:" in file_list_result:
        return f"Failed to list repository files for embedding: {file_list_result}"

    file_paths = file_list_result.split('\n')
    all_documents_for_embedding = []
    
    # Configure the text splitter
    # Adjust chunk_size and chunk_overlap based on typical code file sizes and desired context for LLM
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max characters per chunk
        chunk_overlap=200, # Overlap between chunks to maintain context
        length_function=len,
        add_start_index=True,
    )

    # Limit the number of files to process for very large repos to prevent out-of-memory issues
    max_files_to_process = 200 
    processed_count = 0

    for path in file_paths:
        if processed_count >= max_files_to_process:
            st.warning(f"Stopped processing files after {max_files_to_process} to prevent excessive load. Semantic search will be based on these files.")
            break
        
        # Skip common non-code files or very large files
        if any(path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar.gz', '.bin', '.pdf']) or \
           path.startswith('.git') or path.startswith('node_modules') or path.startswith('__pycache__'):
            continue
        
        # --- MODIFIED LINE FOR THE ATTRIBUTEERROR FIX ---
        # Changed from get_file_content(repo_url, path) back to get_file_content.func(repo_url, path)
        file_content = get_file_content.func(repo_url, path) 
        
        # --- MODIFIED LINE FOR THE "Could not fetch content for app.py" WARNING FIX ---
        # Changed from if "Error:" not in file_content: to if not file_content.startswith("Error:"):
        if not file_content.startswith("Error:"):
        # --- END MODIFIED LINES ---
            # Split the content into chunks
            chunks = text_splitter.create_documents([file_content], metadatas=[{"source": path, "repo_url": repo_url}])
            all_documents_for_embedding.extend(chunks)
            processed_count += 1
        else:
            # This branch will now only execute if get_file_content actually returned a proper error message
            st.warning(f"Could not fetch content for {path}: {file_content}")

    if not all_documents_for_embedding:
        return "No suitable files found or fetched to embed for semantic search."

    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_store = Chroma.from_documents(all_documents_for_embedding, embeddings)
        st.session_state.vector_store_repo_url = repo_url
        return f"Successfully loaded and embedded {len(all_documents_for_embedding)} content chunks from the repository. You can now ask general questions about its content."
    except Exception as e:
        st.error(f"Error creating embeddings or vector store: {e}")
        st.session_state.vector_store = None
        st.session_state.vector_store_repo_url = ""
        return f"Failed to create semantic search index: {e}"

@tool
def query_repository_content(query: str, repo_url: str) -> str:
    """
    Performs a semantic search on the loaded repository content to find relevant information.
    This tool should only be used after 'load_and_embed_repository' has been successfully run for the current URL.
    Args:
        query (str): The question or query to search for within the repository content.
        repo_url (str): The URL of the GitHub repository currently being analyzed.
    Returns:
        str: Relevant text snippets from the repository, or a message indicating no content is loaded.
    """
    # Using st.session_state.vector_store directly for single session
    vector_store = st.session_state.vector_store

    if not vector_store or st.session_state.vector_store_repo_url != repo_url:
        return "Repository content has not been loaded and embedded for semantic search. Please use the 'Load Repository for Semantic Search' button first."

    try:
        # Perform similarity search - will now return relevant chunks
        docs = vector_store.similarity_search(query, k=5) # Retrieve top 5 relevant chunks
        if docs:
            results = []
            for i, doc in enumerate(docs):
                # Indicate that these are snippets/chunks
                results.append(f"--- Relevant Snippet {i+1} from {doc.metadata.get('source', 'unknown file')} (Chunk Start Index: {doc.metadata.get('start_index', 'N/A')}) ---\n")
                results.append(doc.page_content)
                results.append("\n--------------------\n")
            return "\n".join(results)
        else:
            return "No relevant content found for your query in the loaded repository."
    except Exception as e:
        return f"Error during semantic search: {e}"

# --- Langchain Setup ---

def initialize_chatbot(groq_api_key): # Removed current_user_id as it's no longer used for agent init
    """Initializes the Langchain agent with Groq LLM and custom tools."""
    if not groq_api_key:
        return None

    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    tools = [list_repo_files, get_file_content, load_and_embed_repository, query_repository_content]

    # Initialize ConversationBufferMemory for long-term memory
    # Messages will be loaded from st.session_state.messages directly
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Load existing messages for the current session into memory
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are RepoPilot, an expert GitHub repository assistant. Your goal is to help users understand GitHub repositories and solve coding errors.
             If asked for your name, you are RepoPilot.
             You have access to tools to:
             1. List all files in a repository (`list_repo_files`).
             2. Get the content of a specific file (`get_file_content`).
             3. Load and embed the repository content in *chunks* for semantic search (`load_and_embed_repository`).
             4. Query the embedded content *snippets* semantically (`query_repository_content`).

             **Instructions for Error Solving and Code Generation:**
             - If the user provides an error message (e.g., a traceback, a console error), or asks for a code modification/generation:
                 - First, try to identify the relevant file(s) mentioned in the error or implied by the request.
                 - Use `get_file_content` to retrieve the content of the identified file(s). If the user provides a code snippet, analyze that directly.
                 - Analyze the error message in the context of the code.
                 - Propose a solution: This should include an explanation of the problem and the proposed code changes.
                 - **Always output proposed code changes in a markdown code block (e.g., ```python\n# new code\n```).**
                 - If you need more context (e.g., a specific file path, the full error message, or surrounding code), ask the user for it.
                 - If the error seems unrelated to the repository's code (e.g., environment setup), provide general debugging advice.

             **General Instructions:**
             - When a user provides a GitHub URL, first acknowledge it.
             - If the user asks to explain a specific file (e.g., 'explain main.py' or 'what does app.py do?'), you MUST first use the 'get_file_content' tool to retrieve the content of that file. Once you have the content, then provide a detailed explanation.
             - If the user asks a general question about the repository's purpose, functionality, or asks to find information across multiple files (e.g., 'What is the main purpose of this project?', 'How does it handle user authentication?', 'Where are the database interactions?'), first attempt to use the 'query_repository_content' tool with their question.
             - If the 'query_repository_content' tool indicates that the repository content has not been loaded for semantic search, then you MUST instruct the user to use the 'Load Repository for Semantic Search' button.
             - Always try to be as helpful and detailed as possible based on the information you can retrieve, synthesizing information from multiple snippets if needed.
             - If you cannot find a file or repository, or if a tool returns an error, clearly state the error message and explain why you cannot fulfill the request.
             - When asked how to run a project, try to infer from common file names like 'README.md', 'package.json', 'requirements.txt', 'Dockerfile', 'Makefile', etc.
             - If you need more information, ask the user for clarification."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_executor

# --- Helper for File Tree Visualization ---
def build_file_tree_markdown(file_paths_str):
    """Converts a newline-separated string of file paths into a markdown tree structure."""
    if not file_paths_str:
        return "No files found or listed for this repository."

    paths = file_paths_str.split('\n')
    tree = {}

    for path in paths:
        parts = path.split('/')
        current_level = tree
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
    
    def format_tree(node, indent=0):
        markdown = ""
        sorted_keys = sorted(node.keys())
        for i, key in enumerate(sorted_keys):
            prefix = "├── " if i < len(sorted_keys) - 1 else "└── "
            markdown += "    " * indent + prefix + key + "\n"
            if isinstance(node[key], dict) and node[key]:
                markdown += format_tree(node[key], indent + 1)
        return markdown

    return format_tree(tree)


# --- Streamlit Application ---

st.set_page_config(page_title="GitHub Repo Chatbot", layout="wide") # Changed layout to wide for two columns

st.title("🤖 RepoPilot") # Updated title
st.markdown("""
RepoPilot helps you understand GitHub repositories by explaining code files, providing insights on how to run projects, and now, **assisting with code debugging and generation**!
""")

# Sidebar for API keys and instructions
with st.sidebar:
    st.header("Configuration")
    st.markdown("""
    To use this chatbot, you need:
    1.  **Groq API Key**: For the LLM (Large Language Model) inference.
    2.  **GitHub Personal Access Token (Optional but Recommended)**: For higher rate limits and access to private repositories.
        * Go to GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic).
        * Generate a new token with at least `public_repo` scope for public repos, or `repo` for private repos.
    """)
    st.info("It's best to set these as environment variables (`GROQ_API_KEY`, `GITHUB_TOKEN`) or in a `.env` file.")

    # Display current status of keys
    st.subheader("API Key Status:")
    if GROQ_API_KEY:
        st.success("Groq API Key Loaded! ✅")
    else:
        st.error("Groq API Key Missing! ❌")
    if GITHUB_TOKEN:
        st.success("GitHub Token Loaded! ✅")
    else:
        st.warning("GitHub Token Missing (Optional)! ⚠️")

# Create two columns for the main content
col1, col2 = st.columns([2, 1]) # Adjust ratio as needed, e.g., 2 for chat, 1 for debug

with col1: # Left column for general repo interaction and chat
    # Input for GitHub Repository URL (back on main page)
    new_repo_url = st.text_input("Enter GitHub Repository URL:", value=st.session_state.repo_url, key="repo_input")

    # If the URL changes, update active_repo_url and clear relevant data for the current user
    if new_repo_url != st.session_state.repo_url:
        st.session_state.repo_url = new_repo_url
        st.session_state.messages = [] # Clear chat history for new repo
        st.session_state.agent_executor = None # Re-initialize agent if URL changes
        st.session_state.vector_store = None # Clear vector store for new repo
        st.session_state.vector_store_repo_url = ""
        st.session_state.debug_response = "" # Clear debug response as well
        st.rerun()

    # Initialize the chatbot agent if not already initialized
    if st.session_state.agent_executor is None and GROQ_API_KEY:
        st.session_state.agent_executor = initialize_chatbot(GROQ_API_KEY)
        if st.session_state.agent_executor:
            st.success("Chatbot initialized! Ask me about the repository. 🚀")
        else:
            st.error("Failed to initialize chatbot. Please check your Groq API key. 😔")

    # Display file tree and semantic search button only if a repo URL is provided
    if st.session_state.repo_url and st.session_state.agent_executor:
        with st.expander("View Repository File Structure"):
            with st.spinner("Loading file list..."):
                file_list_str = list_repo_files.invoke(st.session_state.repo_url)
                if "Error:" not in file_list_str:
                    st.code(build_file_tree_markdown(file_list_str), language="markdown")
                else:
                    st.error(file_list_str)

        if st.button("Load Repository for Semantic Search"):
            with st.spinner("Loading and embedding repository content... This may take a while for large repos. ⏳"):
                result = load_and_embed_repository.invoke(st.session_state.repo_url)
                st.info(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "```" in message["content"]:
                parts = message["content"].split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        lang = "python" # Default language
                        # Attempt to infer language from the first line of the code block
                        first_line_of_code = part.strip().split('\n')[0]
                        if first_line_of_code.lower().startswith("python"):
                            lang = "python"
                            part = part.lstrip("python\n").lstrip("Python\n")
                        elif first_line_of_code.lower().startswith("javascript"):
                            lang = "javascript"
                            part = part.lstrip("javascript\n").lstrip("JavaScript\n")
                        elif first_line_of_code.lower().startswith("html"):
                            lang = "html"
                            part = part.lstrip("html\n").lstrip("HTML\n")
                        elif first_line_of_code.lower().startswith("css"):
                            lang = "css"
                            part = part.lstrip("css\n").lstrip("CSS\n")
                        elif first_line_of_code.lower().startswith("json"):
                            lang = "json"
                            part = part.lstrip("json\n").lstrip("JSON\n")
                        elif first_line_of_code.lower().startswith("bash"):
                            lang = "bash"
                            part = part.lstrip("bash\n").lstrip("Bash\n")
                        elif first_line_of_code.lower().startswith("diff"):
                            lang = "diff"
                            part = part.lstrip("diff\n").lstrip("Diff\n")
                        elif first_line_of_code.lower().startswith("\n"): # If no explicit language, just trim newline
                            part = part[1:]
                        st.code(part, language=lang)
                    else:
                        st.markdown(part)
            else:
                st.markdown(message["content"])

    # React to general user input (main chat input)
    if prompt := st.chat_input("Ask RepoPilot a general question about the repository...", key="general_chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.repo_url:
            with st.chat_message("assistant"):
                st.error("Please enter a GitHub repository URL first. ⚠️")
            st.session_state.messages.append({"role": "assistant", "content": "Please enter a GitHub repository URL first."})
        elif st.session_state.agent_executor:
            with st.chat_message("assistant"):
                with st.spinner("Thinking... 💬"):
                    try:
                        full_prompt = f"Regarding the repository at {st.session_state.repo_url}, {prompt}"
                        response = st.session_state.agent_executor.invoke({"input": full_prompt})
                        assistant_response = response.get("output", "I'm sorry, I couldn't process that request.")
                        st.markdown(assistant_response)
                    except Exception as e:
                        st.error(f"An error occurred while processing your request: {e} 💔")
                        assistant_response = f"An error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            with st.chat_message("assistant"):
                st.warning("Chatbot not initialized. Please ensure your Groq API Key is set. 🔑")
            st.session_state.messages.append({"role": "assistant", "content": "Chatbot not initialized. Please ensure your Groq API Key is set."})

with col2: # Right column for Code Debugging & Generation
    st.markdown("## Code Debugging & Generation 🛠️")
    error_input = st.text_area(
        "Paste your error message or describe what code you need generated/modified:",
        height=300, # Increased height for better visibility
        key="error_input_area",
        help="Provide the full error traceback, relevant code snippets, and describe what you're trying to achieve."
    )
    if st.button("Solve Error / Generate Code", key="solve_error_button_col2"):
        if error_input:
            # We don't add this to st.session_state.messages, as we want its response in col2
            # Add a temporary user message if you want to see the prompt in the main chat for context,
            # but its response won't be there.
            # st.session_state.messages.append({"role": "user", "content": f"Error/Code Request: {error_input}"})
            
            with st.spinner("Analyzing error and generating solution... 💡"):
                try:
                    # Construct prompt for the agent
                    full_prompt_for_error = f"Regarding the repository at {st.session_state.repo_url}, I encountered this issue/need this code: {error_input}"
                    
                    # Invoke the agent executor
                    response = st.session_state.agent_executor.invoke({"input": full_prompt_for_error})
                    
                    # Store the response specifically for debug/generation output
                    st.session_state.debug_response = response.get("output", "I'm sorry, I couldn't process that request.")
                    
                except Exception as e:
                    st.session_state.debug_response = f"An error occurred while processing your request: {e} 💔"
        else:
            st.session_state.debug_response = "Please paste an error message or describe your code generation/modification request. 📋"
        # Rerun to display the stored debug_response
        st.rerun() 

    # Display the stored debug/generation response below the input area in col2
    if st.session_state.debug_response:
        st.markdown("### RepoPilot's Analysis & Solution:")
        if "```" in st.session_state.debug_response:
            parts = st.session_state.debug_response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    lang = "python" # Default language for code blocks
                    # Attempt to infer language from the first line of the code block
                    first_line_of_code = part.strip().split('\n')[0]
                    if first_line_of_code.lower().startswith("python"):
                        lang = "python"
                        part = part.lstrip("python\n").lstrip("Python\n")
                    elif first_line_of_code.lower().startswith("javascript"):
                        lang = "javascript"
                        part = part.lstrip("javascript\n").lstrip("JavaScript\n")
                    elif first_line_of_code.lower().startswith("html"):
                        lang = "html"
                        part = part.lstrip("html\n").lstrip("HTML\n")
                    elif first_line_of_code.lower().startswith("css"):
                        lang = "css"
                        part = part.lstrip("css\n").lstrip("CSS\n")
                    elif first_line_of_code.lower().startswith("json"):
                        lang = "json"
                        part = part.lstrip("json\n").lstrip("JSON\n")
                    elif first_line_of_code.lower().startswith("bash"):
                        lang = "bash"
                        part = part.lstrip("bash\n").lstrip("Bash\n")
                    elif first_line_of_code.lower().startswith("diff"):
                        lang = "diff"
                        part = part.lstrip("diff\n").lstrip("Diff\n")
                    elif first_line_of_code.lower().startswith("\n"): # If no explicit language, just trim newline
                        part = part[1:]
                    st.code(part, language=lang)
                else:
                    st.markdown(part)
        else:
            st.markdown(st.session_state.debug_response)
