import logging
import os
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
    
    # Define dataset paths to check
    dataset_paths = [
        "Dataset/flood-control-projects-table_2025-09-20.csv",
        "./Dataset/flood-control-projects-table_2025-09-20.csv",
        os.path.join(os.path.dirname(__file__), "Dataset", "flood-control-projects-table_2025-09-20.csv"),
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

# Components
def sidebar():
    with st.sidebar:
        st.markdown("### 📊 Dataset")
        
        # Initialize data handler if not exists
        if 'data_handler' not in st.session_state:
            st.session_state.data_handler = FloodControlDataHandler()
        
        # Initialize data_loaded if not exists
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            
        # Check if LLM is available
        if 'llm_handler' not in st.session_state or not st.session_state.llm_handler:
            st.warning("AI model is not available. Some features may be limited.")
            if st.button("Retry Loading AI Model"):
                try:
                    with st.spinner("Loading AI model..."):
                        st.session_state.llm_handler = LLMHandler(prefer_local=True)
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load AI model: {str(e)}")
            
        try:
            if st.session_state.data_loaded:
                # Verify data_handler has data before calling get_summary_stats
                if hasattr(st.session_state.data_handler, 'df') and st.session_state.data_handler.df is not None:
                    try:
                        stats = st.session_state.data_handler.get_summary_stats()
                        st.markdown(
                            f"<div class='status success'>✅ Loaded {stats.get('total_records', 0):,} records<br>"
                            f"📋 {len(stats.get('columns', []))} columns<br>"
                            f"🌏 {stats.get('unique_regions', 0)} regions<br>"
                            f"📅 {stats.get('date_range', 'N/A')}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.markdown(f"<div class='status error'>❌ Error getting stats: {str(e)[:100]}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='status warn'>⚠️ No data available</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status warn'>⚠️ Dataset not loaded</div>", unsafe_allow_html=True)
                uploaded = st.file_uploader("Upload CSV", type=['csv'])
                if uploaded and st.button("Load"):
                    try:
                        if st.session_state.data_handler.load_csv(uploaded):
                            st.session_state.data_loaded = True
                            st.rerun()
                        else:
                            st.error("Failed to load the uploaded file.")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                        st.session_state.data_loaded = False

            st.markdown("### 🤖 AI")
            if 'llm_handler' in st.session_state:
                try:
                    if st.session_state.llm_handler.is_available():
                        st.markdown("<div class='status success'>✅ AI Model Ready</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='status warn'>⚠️ Basic mode (no AI)</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown("<div class='status error'>❌ AI Service Error</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status warn'>⚠️ AI not initialized</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

def chat_ui():
    """Display the chat interface and handle user interactions."""
    st.subheader("💬 Ask About Flood Control Projects")
    
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
                    results = st.session_state.data_handler.search_relevant_records(prompt, limit=num_results)
                    
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
    
    # Show a warning about the free tier limitations
    if 'show_free_tier_warning' not in st.session_state:
        st.session_state.show_free_tier_warning = True
    
    if st.session_state.show_free_tier_warning:
        with st.sidebar:
            with st.expander("⚠️ Free Tier Notice", expanded=True):
                st.warning("""
                **Free Tier Limitations:**
                - Using GPT-3.5-turbo model
                - Limited to ~3 requests per minute
                - Responses may be shorter to save tokens
                - Some features may be rate-limited
                
                For full capabilities, consider upgrading your OpenAI account.
                """)
                if st.button("I understand", key="dismiss_warning"):
                    st.session_state.show_free_tier_warning = False
                    st.rerun()
        
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
    import streamlit as st
    
    # Check if running in Streamlit Cloud
    IS_STREAMLIT_CLOUD = 'STREAMLIT_SERVER_RUN_ON_UPDATE' in os.environ
    
    if not IS_STREAMLIT_CLOUD:
        # Local development settings
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['BROWSER'] = 'false'
        
        # Configure server settings for local
        st.config.set_option('server.headless', True)
        st.config.set_option('server.enableCORS', True)
        st.config.set_option('server.enableXsrfProtection', True)
        st.config.set_option('server.fileWatcherType', 'none')
        st.config.set_option('server.port', 8501)
    else:
        # Streamlit Cloud specific settings
        st.config.set_option('server.address', '0.0.0.0')
        
    # Disable warning about file downloader
    st.set_option('deprecation.showfileUploaderEncoding', False)

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
        
        # Try to load dataset automatically if not already loaded
        if not st.session_state.get('data_loaded', False):
            if not load_dataset(st.session_state.data_handler, st.session_state.get('logger', logging.getLogger(__name__))):
                st.warning("Please load the dataset using the sidebar.")
        
        # Main layout
        st.title("🌊 Flood Control Projects Assistant")
        st.caption("Ask me about flood control projects in the Philippines")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant helps you explore flood control projects in the Philippines.
        Ask questions like:
        - What's the most expensive project in Cebu?
        - Show me projects in Metro Manila
        - Which contractor has the most projects?
        """)
        
        st.markdown("---")
        st.markdown("### Data Management")
        
        # Upload new dataset
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                if st.session_state.data_handler.load_csv(uploaded_file):
                    st.session_state.data_loaded = True
                    st.success("Dataset loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Clear chat history
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
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
