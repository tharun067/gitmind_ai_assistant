import streamlit as st
import os
import shutil
from src.vector_store_git_loader import (
    EmbeddingManager,
    VectorStoreManager,
    GitDocument,
    EMBEDDING_MODEL_NAME,
    CHROMADB_DIR,
    COLLECTION_NAME
)

from src.llm import agent_build


def clear_database():
    """Remove the ChromaDB directory to start fresh."""
    try:
        messages = []
        success = True
        
        # Clear ChromaDB
        if os.path.exists(CHROMADB_DIR):
            shutil.rmtree(CHROMADB_DIR)
            messages.append("Database cleared successfully!")
        else:
            messages.append("No database found to clear.")
        
        # Clear cloned repository if it exists
        if 'local_path' in st.session_state and st.session_state.local_path:
            local_path = st.session_state.local_path
            if os.path.exists(local_path):
                try:
                    shutil.rmtree(local_path)
                    messages.append(f"Cloned repository at '{local_path}' removed!")
                except Exception as e:
                    messages.append(f"Warning: Could not remove repository at '{local_path}': {e}")
                    success = True  # Don't fail the whole operation
        
        return success, " | ".join(messages)
    except Exception as e:
        return False, f"Error during cleanup: {e}"


def update_status_callback(message, status_container=None):
    """Callback function to update status in Streamlit"""
    if 'status_messages' not in st.session_state:
        st.session_state.status_messages = []
    st.session_state.status_messages.append(message)
    
    if 'progress_bar' in st.session_state:
        progress_bar = st.session_state.progress_bar
        
        if "Loading embedding model" in message:
            progress_bar.progress(10)
        elif "Embedding model loaded" in message:
            progress_bar.progress(20)
        elif "Initializing ChromaDB" in message:
            progress_bar.progress(30)
        elif "Collection" in message and "initialized" in message:
            progress_bar.progress(40)
        elif "Cloning repository" in message or "pulling latest" in message:
            progress_bar.progress(50)
        elif "Repository" in message and "successfully" in message:
            progress_bar.progress(60)
        elif "Loading documents" in message:
            progress_bar.progress(65)
        elif "documents loaded" in message:
            progress_bar.progress(70)
        elif "Splitting documents" in message:
            progress_bar.progress(75)
        elif "text chunks created" in message:
            progress_bar.progress(80)
        elif "Generating embeddings" in message:
            progress_bar.progress(85)
        elif "Embeddings generated" in message:
            progress_bar.progress(90)
        elif "Storing" in message and "documents" in message:
            progress_bar.progress(95)
        elif "All documents processed" in message:
            progress_bar.progress(100)


def initialize_session_state():
    """Initialize all session state variables"""
    if 'conversation_messages' not in st.session_state:
        st.session_state.conversation_messages = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'agent_graph' not in st.session_state:
        st.session_state.agent_graph = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'status_messages' not in st.session_state:
        st.session_state.status_messages = []


def render_sidebar():
    """Render the sidebar configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        repo_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/user_name/repo_name.git",
            help="Enter the Git repository URL",
            key="repo_url"
        )

        local_path = st.text_input(
            "Local Path",
            placeholder="./repo_name",
            help="Local directory to clone the repository",
            key="local_path"
        )

        # ADD THIS NEW SECTION
        groq_api_key = st.text_input(
            "Groq API Key",
            value="",
            type="password",
            help="Enter your Groq API key",
            key="groq_api_key"
        )
        
        st.divider()
        
        st.subheader("📊 Status")
        if st.session_state.processing_complete:
            st.success("✅ Repository processed!")
            st.info("💬 Agent is ready to chat!")
            
            if st.session_state.vector_store:
                try:
                    count = st.session_state.vector_store.collection.count()
                    st.metric("Documents in DB", count)
                except:
                    pass
        else:
            st.warning("⚠️ Process repository first")
        
        st.divider()
        
        if st.session_state.processing_complete:
            st.subheader("🤖 Agent Capabilities")
            st.markdown("""
            - 💻 **Code Agent**: Code analysis & implementation
            - 📚 **Documentation Agent**: Docs & guides
            - 🔍 **QA Agent**: General questions
            - 📝 **History Agent**: Conversation recall
            """)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.conversation_messages = []
                st.session_state.chat_history = []
                st.success("Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                # Clear the database
                success, message = clear_database()
                
                if success:
                    st.info(f"🗑️ {message}")
                else:
                    st.warning(f"⚠️ {message}")
                
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                st.success("✅ All data reset! Please reprocess your repository.")
                st.rerun()
        
        return repo_url, local_path


def render_process_tab():
    """Render the Process Repository tab"""
    st.header("📥 Process Repository")
    st.markdown("Load and process documents from your Git repository into a vector database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Click the button below to start processing. This will clone/update the repository, "
                "load documents, generate embeddings, and store them in ChromaDB.")
    
    with col2:
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
    
    if process_button:
        st.session_state.status_messages = []
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        st.session_state.progress_bar = progress_placeholder.progress(0)
        
        try:
            repo_url = st.session_state.repo_url
            local_path = st.session_state.local_path
            
            with st.spinner("Initializing components..."):
                embedding_manager = EmbeddingManager(
                    model_name=EMBEDDING_MODEL_NAME,
                    status_callback=update_status_callback
                )
                
                vector_store = VectorStoreManager(
                    collection_name=COLLECTION_NAME,
                    db_dir=CHROMADB_DIR,
                    status_callback=update_status_callback
                )
            
            with st.spinner("Processing repository..."):
                pipeline = GitDocument(
                    repo_url=repo_url,
                    local_path=local_path,
                    vector_store=vector_store,
                    embedding_manager=embedding_manager,
                    status_callback=update_status_callback
                )
                
                pipeline.process_and_store_documents()
            
            st.session_state.vector_store = vector_store
            st.session_state.processing_complete = True
            
            update_status_callback("🤖 Initializing AI Agent Graph...")
            try:
                # Pass the API key to agent_build
                groq_key = st.session_state.groq_api_key
                if not groq_key:
                    raise ValueError("Groq API key is required")
    
                st.session_state.agent_graph = agent_build(api_key=groq_key)
                update_status_callback("✅ AI Agent initialized with 4 specialized agents!")
            except Exception as agent_error:
                update_status_callback(f"⚠️ Agent initialization error: {str(agent_error)}")
                st.error(f"Agent initialization failed: {str(agent_error)}")
            
            st.success("✅ Processing completed! Switch to the Chat tab to interact with the agent.")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            if 'progress_bar' in st.session_state:
                del st.session_state.progress_bar
    
    if st.session_state.status_messages:
        st.divider()
        st.subheader("📋 Processing Log")
        
        with st.expander("View detailed log", expanded=True):
            log_text = "\n".join(st.session_state.status_messages)
            st.code(log_text, language=None)


def render_chat_tab():
    """Render the Chat with Agent tab"""
    st.header("💬 Chat with Your Repository Agent")
    
    if not st.session_state.processing_complete:
        st.warning("⚠️ Please process the repository first in the 'Process Repository' tab!")
        st.info("👈 Go to the **Process Repository** tab to get started.")
        
        st.markdown("---")
        st.subheader("🤖 What the Agent Can Do")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **💻 Code Agent**
            - Analyze code structure
            - Find specific functions
            - Explain implementations
            - Show code snippets
            """)
            
            st.markdown("""
            **🔍 QA Agent**
            - Answer general questions
            - Explain concepts
            - Provide summaries
            - Technical Q&A
            """)
        
        with col2:
            st.markdown("""
            **📚 Documentation Agent**
            - Find documentation
            - Explain guides
            - API references
            - How-to instructions
            """)
            
            st.markdown("""
            **📝 History Agent**
            - Recall previous conversations
            - Reference earlier topics
            - Context-aware responses
            - Conversation continuity
            """)
        
        return
    
    st.markdown("Ask questions about your repository. The AI agent will automatically route to the right specialist.")
    
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("💡 Start a conversation by typing a message below or click one of the quick question buttons.")
        
        for role, message in st.session_state.chat_history:
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(message)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message)
    
    user_input = st.chat_input("Type your message here...")
    
    if 'example_query' in st.session_state:
        user_input = st.session_state.example_query
        del st.session_state.example_query
    
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.conversation_messages.append(("user", user_input))
        
        if len(st.session_state.conversation_messages) > 10:
            st.session_state.conversation_messages = st.session_state.conversation_messages[-10:]
        
        try:
            if st.session_state.agent_graph:
                with st.spinner("🤔 Agent is thinking..."):
                    response = st.session_state.agent_graph.invoke({
                        "messages": st.session_state.conversation_messages,
                        "conversation_history": st.session_state.conversation_messages
                    })
                    
                    last_message = response["messages"][-1]
                    if isinstance(last_message, tuple):
                        answer = last_message[1]
                    elif hasattr(last_message, "content"):
                        answer = last_message.content
                    else:
                        answer = str(last_message)
            else:
                answer = "❌ Agent is not initialized. Please reprocess the repository."
            
            st.session_state.conversation_messages.append(("assistant", answer))
            st.session_state.chat_history.append(("assistant", answer))
            
            if len(st.session_state.conversation_messages) > 10:
                st.session_state.conversation_messages = st.session_state.conversation_messages[-10:]
            
            st.rerun()
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            
            with st.expander("Show error details"):
                import traceback
                st.code(traceback.format_exc())
            
            if st.session_state.conversation_messages and st.session_state.conversation_messages[-1][0] == "user":
                st.session_state.conversation_messages.pop()
            if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
                st.session_state.chat_history.pop()
    
    st.divider()
    st.subheader("💡 Quick Questions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📖 Repository Overview", use_container_width=True):
            st.session_state.example_query = "What is this repository about? Give me a brief overview."
            st.rerun()
    
    with col2:
        if st.button("💻 Show Code Structure", use_container_width=True):
            st.session_state.example_query = "Show me the main code structure and key functions."
            st.rerun()
    
    with col3:
        if st.button("📚 Find Documentation", use_container_width=True):
            st.session_state.example_query = "What documentation is available in this repository?"
            st.rerun()
    
    with col4:
        if st.button("🔍 Main Features", use_container_width=True):
            st.session_state.example_query = "What are the main features and capabilities?"
            st.rerun()
    
    st.divider()
    with st.expander("📝 More Example Questions by Agent Type"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💻 Code Agent Examples:**")
            st.markdown("""
            - Show me the main function implementations
            - How is the authentication code structured?
            - Find the API endpoint definitions
            - Explain the database models
            """)
            
            st.markdown("**🔍 QA Agent Examples:**")
            st.markdown("""
            - What technologies are used?
            - How do I set this up?
            - What are the dependencies?
            - Explain the architecture
            """)
        
        with col2:
            st.markdown("**📚 Documentation Agent Examples:**")
            st.markdown("""
            - Show me the API documentation
            - Find the setup guide
            - What's in the README?
            - Explain the configuration options
            """)
            
            st.markdown("**📝 History Agent Examples:**")
            st.markdown("""
            - What did we discuss earlier?
            - Remind me about the previous topic
            - Recall our last conversation
            - What was that function you mentioned?
            """)


def main():
    """Main application function"""
    
    st.set_page_config(
        page_title="Git Repository AI Agent",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("📚 Git Repository AI Agent")
    st.markdown("Process and chat with any Git repository using AI-powered multi-agent system")
    
    repo_url, local_path = render_sidebar()
    
    tab1, tab2 = st.tabs(["📥 Process Repository", "💬 Chat with Agent"])
    
    with tab1:
        render_process_tab()
    
    with tab2:
        render_chat_tab()
    
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<p style='text-align: center; color: gray;'>"
            "Built with Streamlit • LangChain • LangGraph • ChromaDB • Groq"
            "</p>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()