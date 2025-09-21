import logging
import os
import re
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the project modules
sys.path.append(str(Path(__file__).parent.absolute()))

import streamlit as st
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(handler=None, logger=None):
    """Try to load the dataset from common locations."""
    if handler is None:
        if 'data_handler' in st.session_state:
            handler = st.session_state.data_handler
        else:
            return False
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if we're running on Streamlit Cloud
    is_streamlit_cloud = 'STREAMLIT_SERVER_RUNNING_REMOTELY' in os.environ
    
    # Define dataset paths to check (prioritize the exact file mentioned by user)
    dataset_paths = [
        os.path.join(os.path.dirname(__file__), "Dataset", "flood-control-projects-table_2025-09-20.csv"),
        "Dataset/flood-control-projects-table_2025-09-20.csv",
        "./Dataset/flood-control-projects-table_2025-09-20.csv",
    ]
    
    # If running on Streamlit Cloud, try to load from the data directory first
    if is_streamlit_cloud:
        cloud_path = os.path.join(os.path.dirname(__file__), "data", "flood-control-projects-table_2025-09-20.csv")
        if os.path.exists(cloud_path):
            dataset_paths.insert(0, cloud_path)
    
    # Try each path
    for path in dataset_paths:
        try:
            if os.path.exists(path):
                if handler.load_csv_from_path(path):
                    logger.info(f"Successfully loaded dataset from {path}")
                    st.session_state.data_loaded = True
                    return True
        except Exception as e:
            logger.error(f"Error loading dataset from {path}: {str(e)}")
            continue
    
    # If we get here, no dataset was found
    if is_streamlit_cloud:
        st.error("""
        ❌ Dataset not found in the expected location. Please ensure you've added the dataset file to the app.
        
        On Streamlit Cloud, you can upload your dataset file (flood-control-projects-table_2025-09-20.csv) 
        to the `data/` directory in your repository.
        """)
    else:
        st.error("""
        ❌ Dataset not found. Please ensure the dataset file exists in one of these locations:
        - Dataset/flood-control-projects-table_2025-09-20.csv
        - ./Dataset/flood-control-projects-table_2025-09-20.csv
        - data/flood-control-projects-table_2025-09-20.csv (for Streamlit Cloud)
        """)
    
    return False

# Components (old sidebar function removed - now integrated into main)

def chat_ui():
    """Display the chat interface and handle user interactions."""
    
    # Only show the chat interface if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.info("🔄 Please wait while we load the flood control projects dataset...")
        return
    
    # Show welcome message if no chat history
    if not st.session_state.get('chat_history', []):
        with st.chat_message("assistant"):
            st.markdown("""
            👋 **Welcome to FloodWise PH!** 
            
            I'm here to help you explore flood control projects across the Philippines. I have access to thousands of DPWH project records.
            
            **Try asking me:**
            - "What are the most expensive projects in Cebu?"
            - "Show me recent flood control projects in Metro Manila"
            - "Which contractors are most active in Region VII?"
            - "Tell me about drainage projects completed in 2023"
            
            What would you like to know? 🌊
            """)
    
    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    
    # Chat input
    if prompt := st.chat_input("Ask about flood control projects..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Parse the query for number of results requested
                    num_match = re.search(r'(?:show|list|top|first|get)\s+(?:me\s+)?(\d+)(?:\s+results?)?', prompt, re.IGNORECASE)
                    num_results = int(num_match.group(1)) if num_match else 5
                    
                    # Search for relevant records
                    results = st.session_state.data_handler.search_relevant_records(prompt, top_k=num_results)
                    
                    # Generate response
                    response = st.session_state.llm_handler.generate_response(
                        query=prompt,
                        results=results
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

def footer():
    st.markdown(
        "<footer>🌊 FloodWise PH • 🇵🇭 Philippines Flood Control Intelligence<br>"
        "🚀 Built with Streamlit • 🤖 AI-powered • 📱 Mobile Ready</footer>",
        unsafe_allow_html=True,
    )

def initialize_handlers():
    """Initialize data and LLM handlers with error handling."""
    # Initialize data handler if not exists
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = FloodControlDataHandler()
    
    # Set data_loaded flag if not exists
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Show a compact notice about the free tier limitations (no interaction required)
    if 'show_free_tier_warning' not in st.session_state:
        st.session_state.show_free_tier_warning = False  # Auto-dismiss
        
    # Initialize LLM handler if not exists
    if 'llm_handler' not in st.session_state:
        try:
            with st.spinner("Initializing AI model..."):
                # Check for API key in multiple locations
                api_key = None
                
                # 1. Check Streamlit secrets
                if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                    api_key = st.secrets['OPENAI_API_KEY']
                # 2. Check environment variables
                elif 'OPENAI_API_KEY' in os.environ:
                    api_key = os.environ['OPENAI_API_KEY']
                # 3. Check .env file
                else:
                    try:
                        from dotenv import load_dotenv
                        load_dotenv()
                        api_key = os.getenv('OPENAI_API_KEY')
                    except:
                        pass
                
                if not api_key:
                    st.warning("""
                    ⚠️ **OpenAI API Key Not Found**
                    
                    To use this app, you'll need an OpenAI API key. Here's how to get started:
                    
                    1. Sign up at [OpenAI](https://platform.openai.com/signup)
                    2. Get your API key from the [API Keys](https://platform.openai.com/account/api-keys) page
                    3. Add it to your Streamlit secrets or as an environment variable
                    
                    The app will use a limited fallback mode without the API key.
                    """)
                    st.session_state.llm_handler = None
                    return
                
                # Set the API key in environment if not already set
                if 'OPENAI_API_KEY' not in os.environ:
                    os.environ['OPENAI_API_KEY'] = api_key
                
                # Initialize the LLM handler with gpt-3.5-turbo (free tier compatible)
                st.session_state.llm_handler = LLMHandler(model_name="gpt-3.5-turbo")
                
                # Test if the model is available
                if not st.session_state.llm_handler.is_available():
                    st.warning("""
                    ⚠️ **AI Model Not Available**
                    
                    The AI model couldn't be loaded. This might be because:
                    - Your API key is invalid or has expired
                    - You've exceeded your usage limits
                    - There's a temporary issue with the API
                    
                    The app will use a limited fallback mode.
                    """)
                    
        except Exception as e:
            st.error(f"❌ Error initializing AI model: {str(e)}")
            st.session_state.llm_handler = None
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def check_system_resources():
    """Check if system has enough resources to run the app."""
    try:
        import psutil
        import shutil
        
        # Check available RAM (in MB)
        available_ram = psutil.virtual_memory().available / (1024 * 1024)
        if available_ram < 2000:  # Less than 2GB
            st.warning(f"⚠️ Low memory available: {available_ram:.0f}MB. The app might be slow or crash.")
            
        # Check disk space (in GB)
        _, _, free_disk = shutil.disk_usage("/")
        free_disk_gb = free_disk / (1024 * 1024 * 1024)
        if free_disk_gb < 2:  # Less than 2GB free
            st.warning(f"⚠️ Low disk space: {free_disk_gb:.1f}GB free. The app might not work properly.")
            
    except Exception as e:
        logger.warning(f"Could not check system resources: {e}")

def configure_streamlit():
    """Configure Streamlit settings for both local and cloud environments."""
    import os
    
    # Check if running in Streamlit Cloud
    IS_STREAMLIT_CLOUD = 'STREAMLIT_SERVER_RUN_ON_UPDATE' in os.environ
    
    if not IS_STREAMLIT_CLOUD:
        # Local development settings
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['BROWSER'] = 'false'
        
        # Only set configurations that are still supported
        try:
            import streamlit as st
            # These configurations may not be available in all Streamlit versions
            # Wrap in try-except to prevent errors
            try:
                st.config.set_option('server.headless', True)
            except:
                pass
            try:
                st.config.set_option('server.enableCORS', True)
            except:
                pass
            try:
                st.config.set_option('server.fileWatcherType', 'none')
            except:
                pass
        except Exception as e:
            logger.warning(f"Could not configure Streamlit settings: {e}")
    
    # Note: Removed server.address setting as it may cause issues in some environments

def main():
    """Main application function."""
    try:
        # Configure Streamlit first
        configure_streamlit()
        
        # Set page config
        st.set_page_config(
            page_title="FloodWise PH",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add custom CSS for better mobile experience
        st.markdown("""
        <style>
            /* Make the chat container more compact */
            .stChatFloatingInputContainer {
                max-width: 900px;
                margin: 0 auto;
            }
            
            /* Better spacing for mobile */
            @media (max-width: 768px) {
                .stChatFloatingInputContainer {
                    padding: 10px;
                }
                
                /* Hide the sidebar on mobile for more space */
                section[data-testid="stSidebar"] {
                    display: none;
                }
                
                /* Make the main content full width on mobile */
                .main .block-container {
                    padding: 1rem;
                }
                
                /* Better spacing for chat messages */
                .stChatMessage {
                    padding: 0.5rem;
                }
            }
            
            /* Style for warning messages */
            .stAlert {
                border-left: 4px solid #f4c430;
                padding: 0.5rem 1rem;
                margin: 1rem 0;
                border-radius: 0.25rem;
                background-color: rgba(244, 196, 48, 0.1);
            }
            
            /* Style for error messages */
            .stException {
                border-left: 4px solid #ff4b4b;
                padding: 0.5rem 1rem;
                margin: 1rem 0;
                border-radius: 0.25rem;
                background-color: rgba(255, 75, 75, 0.1);
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Check system resources
        check_system_resources()
        
        # Initialize with error handling
        initialize_handlers()
        
        # Automatically load dataset if not already loaded
        if not st.session_state.get('data_loaded', False):
            with st.spinner("Loading flood control projects dataset..."):
                if load_dataset(st.session_state.data_handler, logging.getLogger(__name__)):
                    # Get dataset stats for the success message
                    try:
                        stats = st.session_state.data_handler.get_summary_stats()
                        st.success(f"✅ Loaded {stats.get('total_records', 0):,} flood control projects from DPWH records!")
                        
                        # Add debug info in sidebar
                        with st.sidebar:
                            with st.expander("🔍 Debug Info", expanded=False):
                                st.write(f"**Columns:** {len(stats.get('columns', []))}")
                                st.write(f"**Sample columns:** {stats.get('columns', [])[:5]}")
                                if st.session_state.data_handler.df is not None:
                                    sample_contractor = st.session_state.data_handler.df['Contractor'].dropna().iloc[0] if 'Contractor' in st.session_state.data_handler.df.columns else 'N/A'
                                    sample_project = st.session_state.data_handler.df['ProjectDescription'].dropna().iloc[0] if 'ProjectDescription' in st.session_state.data_handler.df.columns else 'N/A'
                                    st.write(f"**Sample Contractor:** {sample_contractor}")
                                    st.write(f"**Sample Project:** {sample_project[:50]}...")
                    except Exception as e:
                        st.success("✅ Dataset loaded successfully! You can now ask questions about flood control projects.")
                        st.error(f"Debug: Error getting stats: {e}")
                else:
                    st.error("❌ Could not load the dataset automatically. Please check if the dataset file exists.")
        
        # Main layout
        st.title("🌊 Flood Control Projects Assistant")
        st.caption("Ask me about flood control projects in the Philippines")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🌊 FloodWise PH")
        
        # Show dataset status
        if st.session_state.get('data_loaded', False):
            try:
                stats = st.session_state.data_handler.get_summary_stats()
                st.success(f"📊 {stats.get('total_records', 0):,} projects loaded")
                st.info(f"🌏 {stats.get('unique_regions', 0)} regions covered")
            except:
                st.success("✅ Dataset loaded")
        else:
            st.warning("⏳ Loading dataset...")
        
        st.markdown("---")
        st.markdown("### 💬 Sample Questions")
        st.markdown("""
        Try asking:
        - *"What's the most expensive project in Cebu?"*
        - *"Show me projects in Metro Manila"*
        - *"Which contractor has the most projects?"*
        - *"What flood control projects were completed in 2023?"*
        - *"Tell me about drainage projects in Davao"*
        """)
        
        st.markdown("---")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
            
        # Show AI status
        st.markdown("### 🤖 AI Status")
        if st.session_state.get('llm_handler') and st.session_state.llm_handler.is_available():
            st.success("✅ AI Ready")
        else:
            st.warning("⚠️ Basic Mode")
    
    # Display chat interface
    chat_ui()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2em;'>"
        "🌊 FloodWise PH • 🇵🇭 Philippines Flood Control Intelligence<br>"
        "🚀 Built with Streamlit • 🤖 AI-powered • 📱 Mobile Ready"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("""
            ## ⚠️ Critical Error
            The application encountered an unexpected error and needs to close.
            Please try refreshing the page. If the problem persists, please report this issue.
            \n\n**Error Details:**\n""")
        st.code(f"{str(e)}\n\n{traceback.format_exc()}")
        st.stop()
