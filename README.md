
# 🤖 RepoPilot: Your AI-Powered GitHub Repository Assistant

RepoPilot is an intelligent chatbot designed to streamline your interaction with GitHub repositories. Powered by large language models and advanced retrieval techniques, it acts as your personal AI assistant for understanding codebases, debugging issues, and even generating code snippets. It's specifically tailored to help developers building GenAI and Agentic AI applications.



https://github.com/user-attachments/assets/f3998c84-ef3d-46c7-ad6f-77af0aab1c7c



## ✨ Features
GitHub Repository Integration: Easily connect to any public GitHub repository by providing its URL.

Semantic Search (RAG): Load and embed repository content into a vector store to perform semantic searches. Ask high-level questions about the codebase's purpose, architecture, or specific functionalities.

File Tree Visualization: Get a quick overview of the repository's structure with an interactive file tree

Specialized Code Debugging & Error Solving:

Dedicated section for pasting error messages (full tracebacks, console errors) or describing coding problems.

RepoPilot analyzes the error in the context of the loaded repository code (if available) and provides proposed solutions, including modified code snippets.

Expertise in GenAI/Agentic AI: Specializes in diagnosing issues related to LLM API errors, prompt engineering, tool/agent execution, vector database problems, and data handling in AI applications.

Code Generation: Request new code snippets or modifications based on your descriptions, with output formatted in clear code blocks.

Conversational Memory: RepoPilot remembers the context of your conversation, allowing for more natural and continuous interactions.
Named AI Assistant: When asked, RepoPilot proudly identifies itself by its name.

<img width="1167" height="886" alt="RepoPilot New Diagram" src="https://github.com/user-attachments/assets/d2002057-1b48-436a-84f9-1181185a8bf1" />

## 🚀 Getting Started
Follow these steps to set up and run RepoPilot locally.

Prerequisites
Python 3.9+

pip (Python package installer)

1. Clone the Repository
First, clone this GitHub repository to your local machine:

    git clone https://github.com/Mr-aicoder/RepoPilot.git
    cd RepoPilot # Navigate into the project directory

2. Set up Virtual Environment (Recommended)
It's highly recommended to use a virtual environment to manage dependencies:

python -m venv venv
# On Windows:
    .\venv\Scripts\activate
# On macOS/Linux:
    source venv/bin/activate

3. Install Dependencies
Install all required Python packages using the requirements.txt file. Make sure your requirements.txt includes:

    streamlit
    langchain-groq
    langchain-core
    langchain
    requests
    python-dotenv
    langchain-community
    sentence-transformers
    chromadb
    pysqlite3 # Added for deployment compatibility

Then run:

    pip install -r requirements.txt

4. Configure API Keys
RepoPilot requires API keys for the LLM (Groq) and GitHub.

Groq API Key: Obtain one from Groq Console.

GitHub Personal Access Token (PAT):

    Go to GitHub Settings -> Developer settings -> Personal access tokens -> Tokens (classic).

Generate a new token with at least public_repo scope for public repositories, or repo scope for private repositories.

Create a file named .env in the root directory of your project (the same directory as app.py) and add your keys:

# .env
    GROQ_API_KEY="sk_your_groq_api_key_here"
    GITHUB_TOKEN="ghp_your_github_token_here"

Important: Ensure .env is in your .gitignore file to prevent accidentally committing your sensitive keys to GitHub.

5. Run the Application
Once everything is set up, run the Streamlit application:

       streamlit run app.py

Your browser should automatically open to the Streamlit app (usually http://localhost:8501 or http://localhost:8502).

## ☁️ Deployment to Streamlit Community Cloud
Deploying RepoPilot to Streamlit Community Cloud makes it accessible via a public URL.

Ensure Public GitHub Repository: Your project must be in a public GitHub repository.

pysqlite3 workaround: The __import__('pysqlite3') and sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') lines at the top of your app.py are crucial for some deployment environments (like Streamlit Community Cloud) that might have issues with the default sqlite3 library used by chromadb. Ensure these lines are present.

Set Secrets: Crucially, you must add your GROQ_API_KEY and GITHUB_TOKEN as secrets directly in the Streamlit Community Cloud dashboard under "Advanced settings" when deploying. Do not rely on the .env file for cloud deployment.

Secret name: GROQ_API_KEY

Value: sk_your_groq_api_key_here (your actual key)

Secret name: GITHUB_TOKEN

Value: ghp_your_github_token_here (your actual token)

Follow the Streamlit Deployment Guide: For detailed step-by-step instructions on deploying, refer to the official Streamlit deployment guide.





## 💡 How to Use RepoPilot
RepoPilot features a two-column layout for an enhanced experience:

Left Column (Main Chat & Repo Overview):

Enter GitHub Repository URL: Paste the URL of the GitHub repository you want to analyze.

Load for Semantic Search: Click the "Load Repository for Semantic Search" button. This processes the repository's files into chunks, enabling advanced semantic queries.

View File Structure: Expand the "View Repository File Structure" section to see a clear, interactive tree-like representation of the codebase.

Ask General Questions: Use the chat input at the bottom of the left column to ask high-level questions about the repository (e.g., "What is the main purpose of this project?", "How does it handle user authentication?").

Right Column (Code Debugging & Generation):

Paste Error/Request: In the dedicated text area, paste your error message (full traceback recommended) or describe the code you need generated/modified.

Solve Error / Generate Code: Click this button. RepoPilot will analyze your input, fetch relevant code from the loaded repository, diagnose the problem (especially for GenAI/Agentic AI issues), and provide a solution or generated code in the left column's chat history.

## 🤝 Contributing
RepoPilot is an open-source project. Contributions are welcome! Feel to open issues for bugs or feature requests, or submit pull requests with improvements.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
Built with Streamlit

Powered by LangChain

LLM provided by Groq

Embeddings from Sentence Transformers

Vector Store by Chroma
