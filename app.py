import os
import re
import logging
import streamlit as st
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler

def load_dataset(handler=None, logger=None):
    """Try to load the dataset from common locations."""
    if handler is None:
        if 'data_handler' in st.session_state:
            handler = st.session_state.data_handler
        else:
            return False
    
    if logger is None:
        logger = logging.getLogger(__name__)
        
    dataset_paths = [
        "Dataset/flood-control-projects-table_2025-09-20.csv",
        "./Dataset/flood-control-projects-table_2025-09-20.csv",
        os.path.join(os.path.dirname(__file__), "Dataset", "flood-control-projects-table_2025-09-20.csv"),
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                if handler.load_csv_from_path(path):
                    logger.info(f"Successfully loaded dataset from {path}")
                    st.session_state.data_loaded = True
                    return True
            except Exception as e:
                logger.error(f"Error loading dataset from {path}: {str(e)}")
                continue
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
        
    # Initialize LLM handler if not exists
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler(prefer_local=True)
        
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # Set data_loaded flag if not exists
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def main():
    """Main application function."""
    # Set page config - MUST be the first Streamlit command
    st.set_page_config(
        page_title="Flood Control Projects Assistant", 
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
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
                return False, None  # Success
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return False, str(e)
        
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
    main()
