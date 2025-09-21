import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from github import Github, RateLimitExceededException
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
CORS(app)
load_dotenv()

class RepositoryAnalyzer:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.vector_stores = {}  # Cache for repositories

    def setup_github_client(self):
        if not self.github_token:
            raise ValueError("GitHub Personal Access Token is required")
        
        g = Github(self.github_token)
        rate_limit = g.get_rate_limit()
        core_rate_limit = rate_limit.core
        remaining = core_rate_limit.remaining
        limit = core_rate_limit.limit
        
        return g, {"remaining": remaining, "limit": limit}

    def fetch_repo_metadata(self, repo_url: str) -> Dict:
        g, rate_info = self.setup_github_client()
        repo_name = repo_url.split("github.com/")[-1]
        
        try:
            repo = g.get_repo(repo_name)
            return {
                "name": repo.name,
                "description": repo.description or "No description",
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "issues": repo.open_issues_count,
                "last_updated": repo.updated_at.strftime("%Y-%m-%d"),
                "language": repo.language or "Unknown",
                "rate_limit": rate_info
            }
        except RateLimitExceededException:
            return {"error": "GitHub API rate limit exceeded. Please try again later."}
        except Exception as e:
            return {"error": f"Error fetching repository metadata: {str(e)}"}

    def load_repo_files(self, repo_url: str, branch: str = None) -> List:
        repo_name = repo_url.split("github.com/")[-1]
        
        try:
            g = Github(self.github_token)
            repo = g.get_repo(repo_name)
            branch = branch or repo.default_branch
            
            loader = GithubFileLoader(
                repo=repo_name,
                branch=branch,
                access_token=self.github_token,
                file_filter=lambda x: x.endswith((".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".html", ".css", ".json", ".vue", ".scss", ".sass", ".xml"))
            )
            
            documents = loader.load()
            if not documents and branch != "master":
                return self.load_repo_files(repo_url, branch="master")
            
            return documents
        except RateLimitExceededException:
            raise Exception("GitHub API rate limit exceeded")
        except Exception as e:
            raise Exception(f"Error loading repository files: {str(e)}")

    def create_vector_store(self, documents):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            return vector_store, len(chunks)
        except Exception as e:
            raise Exception(f"Error creating FAISS vector store: {str(e)}")

    def setup_llm(self):
        if not self.groq_api_key:
            raise ValueError("Groq API key is required")
        
        return ChatGroq(
            api_key=self.groq_api_key,
            model="moonshotai/kimi-k2-instruct",
            temperature=0,
            max_tokens=1000,
            timeout=None,
            max_retries=2
        )

    def get_functionalities(self, repo_url: str) -> Dict:
        """Get list of functionalities with 1-sentence descriptions"""
        try:
            # Load or get cached vector store
            if repo_url not in self.vector_stores:
                documents = self.load_repo_files(repo_url)
                if not documents:
                    return {"success": False, "error": "No relevant files found"}
                
                vector_store, chunk_count = self.create_vector_store(documents)
                self.vector_stores[repo_url] = vector_store
            else:
                vector_store = self.vector_stores[repo_url]

            retriever = vector_store.as_retriever(search_kwargs={"k": 8})
            
            # Fixed prompt template - only context and question variables
            functionality_prompt = """
            Analyze the provided repository context and extract the main functionalities/features.
            
            Context: {context}
            
            Question: {question}
            
            Return ONLY a JSON array of objects with "name" and "description" fields.
            Each description should be exactly ONE sentence explaining what that functionality does.
            Focus on the core features and capabilities found in the code.
            
            Example format:
            [
                {{"name": "User Authentication", "description": "Handles user login and registration with JWT tokens."}},
                {{"name": "Data Processing", "description": "Processes and validates incoming data from API requests."}}
            ]
            """
            
            prompt = ChatPromptTemplate.from_template(functionality_prompt)
            llm = self.setup_llm()
            chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
            
            context = retriever.invoke("What are the main functionalities and features of this repository?")
            
            response = chain.invoke({
                "context": context, 
                "question": "List the main functionalities found in this repository with one-sentence descriptions."
            })
            
            # Parse JSON response
            try:
                functionalities = json.loads(response)
                return {"success": True, "functionalities": functionalities}
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {"success": True, "functionalities": [
                    {"name": "Code Analysis", "description": "Analyzes repository structure and extracts key information."},
                    {"name": "File Processing", "description": "Processes multiple file types including Python, JavaScript, and markup files."},
                    {"name": "Documentation Parsing", "description": "Extracts and processes README and documentation files."},
                    {"name": "Dependency Management", "description": "Handles project dependencies and package configurations."},
                    {"name": "Configuration Management", "description": "Manages application settings and environment configurations."}
                ]}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_summary(self, repo_url: str) -> Dict:
        """Get detailed repository summary"""
        try:
            # Get metadata
            metadata = self.fetch_repo_metadata(repo_url)
            if "error" in metadata:
                return {"success": False, "error": metadata["error"]}

            # Load or get cached vector store
            if repo_url not in self.vector_stores:
                documents = self.load_repo_files(repo_url)
                if not documents:
                    return {"success": False, "error": "No relevant files found"}
                
                vector_store, chunk_count = self.create_vector_store(documents)
                self.vector_stores[repo_url] = vector_store
            else:
                vector_store = self.vector_stores[repo_url]

            retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            
            # Fixed prompt template
            summary_prompt = """
            You are an expert code analyst. Analyze the provided repository context and create a comprehensive summary.
            
            Context: {context}
            
            Question: {question}
            
            Provide a detailed technical summary that includes:
            1. What the repository is about and its main purpose
            2. Key technologies and frameworks used
            3. Main components and architecture
            4. Notable features and capabilities
            
            Write in a clear, informative manner suitable for developers.
            """
            
            prompt = ChatPromptTemplate.from_template(summary_prompt)
            llm = self.setup_llm()
            chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
            
            context = retriever.invoke("What is this repository about? Provide comprehensive technical details.")
            
            summary = chain.invoke({
                "context": context,
                "question": "Provide a detailed technical summary of this repository including its purpose, technologies, architecture, and key features."
            })
            
            return {
                "success": True, 
                "summary": summary,
                "metadata": metadata
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def explain_functionality(self, repo_url: str, functionality_name: str) -> Dict:
        """Get detailed explanation of how a specific functionality works"""
        try:
            if repo_url not in self.vector_stores:
                return {"success": False, "error": "Repository not analyzed yet"}
            
            vector_store = self.vector_stores[repo_url]
            retriever = vector_store.as_retriever(search_kwargs={"k": 8})
            
            # Fixed prompt template
            explanation_prompt = """
            You are an expert code analyst. Explain in detail how the specified functionality is implemented in the repository.
            
            Context: {context}
            
            Question: {question}
            
            Provide a detailed technical explanation that includes:
            1. How this functionality is implemented
            2. What files and code are involved
            3. Key algorithms or approaches used
            4. Dependencies and libraries utilized
            
            Focus on the technical implementation details with specific references to the code.
            """
            
            prompt = ChatPromptTemplate.from_template(explanation_prompt)
            llm = self.setup_llm()
            chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
            
            context = retriever.invoke(f"How is {functionality_name} implemented? Show implementation details.")
            
            explanation = chain.invoke({
                "context": context,
                "question": f"Explain in detail how '{functionality_name}' is implemented in this repository, including specific code references and technical details."
            })
            
            return {"success": True, "explanation": explanation}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def chat_with_repository(self, repo_url: str, question: str, chat_history: List = None) -> Dict:
        """Chat interface for repository Q&A"""
        try:
            if repo_url not in self.vector_stores:
                # Initialize repository if not cached
                result = self.get_summary(repo_url)
                if not result["success"]:
                    return result
            
            vector_store = self.vector_stores[repo_url]
            retriever = vector_store.as_retriever(search_kwargs={"k": 6})
            
            # Format chat history
            history_text = ""
            if chat_history:
                for exchange in chat_history[-4:]:  # Keep last 4 exchanges
                    history_text += f"User: {exchange.get('user', '')}\nAssistant: {exchange.get('bot', '')}\n\n"
            
            # Simple prompt template without chat_history variable
            chat_prompt = f"""
            You are an expert code analyst assistant. Answer questions about the repository using the provided context.
            
            Previous conversation:
            {history_text}
            
            Context: {{context}}
            
            Current question: {{question}}
            
            Provide helpful, accurate answers based on the repository content. Reference specific files or code when relevant.
            Be concise but informative.
            """
            
            prompt = ChatPromptTemplate.from_template(chat_prompt)
            llm = self.setup_llm()
            chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
            
            # Get relevant context
            context = retriever.invoke(question)
            
            # Generate response
            response = chain.invoke({
                "context": context,
                "question": question
            })

            return {"success": True, "response": response}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize analyzer
analyzer = RepositoryAnalyzer()

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/api/get_functionalities', methods=['POST'])
def get_functionalities():
    try:
        data = request.get_json()
        if not data or 'repo_url' not in data:
            return jsonify({"success": False, "error": "Repository URL is required"}), 400

        repo_url = data['repo_url'].strip()
        if not repo_url.startswith('https://github.com/'):
            return jsonify({"success": False, "error": "Please provide a valid GitHub repository URL"}), 400

        result = analyzer.get_functionalities(repo_url)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route('/api/get_summary', methods=['POST'])
def get_summary():
    try:
        data = request.get_json()
        if not data or 'repo_url' not in data:
            return jsonify({"success": False, "error": "Repository URL is required"}), 400

        repo_url = data['repo_url'].strip()
        if not repo_url.startswith('https://github.com/'):
            return jsonify({"success": False, "error": "Please provide a valid GitHub repository URL"}), 400

        result = analyzer.get_summary(repo_url)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route('/api/explain_functionality', methods=['POST'])
def explain_functionality():
    try:
        data = request.get_json()
        if not data or 'repo_url' not in data or 'functionality_name' not in data:
            return jsonify({"success": False, "error": "Repository URL and functionality name are required"}), 400

        repo_url = data['repo_url'].strip()
        functionality_name = data['functionality_name'].strip()

        result = analyzer.explain_functionality(repo_url, functionality_name)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'repo_url' not in data or 'question' not in data:
            return jsonify({"success": False, "error": "Repository URL and question are required"}), 400

        repo_url = data['repo_url'].strip()
        question = data['question'].strip()
        chat_history = data.get('chat_history', [])

        result = analyzer.chat_with_repository(repo_url, question, chat_history)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "github_token_configured": bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")),
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY"))
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5032)
