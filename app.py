import streamlit as st
import pandas as pd
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler
from mobile_utils import optimize_for_mobile
import os
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="FloodWise PH - Philippines Flood Control Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="auto"  # Changed to auto for better mobile experience
)

# Apply mobile optimizations
optimize_for_mobile()

# Mobile-responsive CSS
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        
        /* Hide sidebar by default on mobile */
        .css-1d391kg {
            width: 0px;
        }
        
        /* Adjust main content area for mobile */
        .css-18e3th9 {
            padding-left: 1rem;
        }
        
        /* Make text inputs full width on mobile */
        .stTextInput > div > div > input {
            font-size: 16px; /* Prevents zoom on iOS */
        }
        
        /* Responsive columns */
        .css-1r6slb0 {
            flex-direction: column;
        }
        
        /* Chat messages styling for mobile */
        .chat-message {
            margin-bottom: 1rem;
            padding: 0.5rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
        }
        
        /* Mobile-friendly buttons */
        .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
        
        /* Responsive dataframes */
        .dataframe {
            font-size: 12px;
            overflow-x: auto;
        }
        
        /* Header adjustments for mobile */
        h1 {
            font-size: 1.8rem !important;
            line-height: 1.2;
        }
        
        h2 {
            font-size: 1.4rem !important;
        }
        
        h3 {
            font-size: 1.2rem !important;
        }
    }
    
    /* Tablet styles */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        .css-1d391kg {
            width: 250px;
        }
    }
    
    /* Desktop styles */
    @media (min-width: 1025px) {
        .main .block-container {
            max-width: 1200px;
        }
    }
    
    /* General improvements */
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stButton > button {
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Chat interface styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 2rem;
    }
    
    /* Responsive sidebar */
    .css-1d391kg .css-1v0mbdj {
        padding: 1rem;
    }
    
    /* Loading spinner improvements */
    .stSpinner {
        text-align: center;
    }
    
    /* Mobile navigation improvements */
    @media (max-width: 768px) {
        .css-1v0mbdj {
            padding: 0.5rem;
        }
        
        .sidebar-content {
            font-size: 14px;
        }
        
        /* Collapsible sections on mobile */
        .streamlit-expanderHeader {
            font-size: 14px;
        }
    }
    
    /* Touch-friendly elements */
    .stSelectbox > div > div {
        min-height: 44px; /* Apple's recommended minimum touch target */
    }
    
    .stButton > button {
        min-height: 44px;
        font-size: 16px;
    }
    
    /* Improved form styling */
    .stForm {
        border: none;
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = FloodControlDataHandler()
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = LLMHandler()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Auto-load the dataset on first run
if not st.session_state.data_loaded:
    dataset_path = "Dataset/flood-control-projects-table_2025-09-20.csv"
    if os.path.exists(dataset_path):
        with st.spinner("Loading flood control projects dataset..."):
            if st.session_state.data_handler.load_csv_from_path(dataset_path):
                st.session_state.data_loaded = True

def render_mobile_header():
    """Render mobile-optimized header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌊 FloodWise PH")
        st.markdown("**Philippines Flood Control Intelligence**")
    with col2:
        # Mobile menu toggle
        if st.button("☰ Menu", key="mobile_menu"):
            st.session_state.show_sidebar = not st.session_state.get('show_sidebar', False)

def render_chat_message(message_type, content, index):
    """Render individual chat message with mobile-friendly styling."""
    if message_type == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>Assistant:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

def render_mobile_sidebar():
    """Render mobile-optimized sidebar content."""
    with st.sidebar:
        st.markdown("### 📱 FloodWise PH")
        
        # Data information
        if st.session_state.data_loaded:
            with st.expander("📊 Dataset Info", expanded=False):
                stats = st.session_state.data_handler.get_summary_stats()
                st.markdown(f"""
                <div class="status-success">
                    ✅ <strong>Dataset Loaded</strong><br>
                    📊 {stats['total_records']:,} records<br>
                    📋 {len(stats['columns'])} columns<br>
                    🌏 All PH regions<br>
                    📅 2021-2024 period
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-warning">
                ⚠️ Dataset not loaded
            </div>
            """, unsafe_allow_html=True)
            
            # Fallback upload
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=['csv'],
                help="Upload flood control project data"
            )
            
            if uploaded_file and st.button("Load Data", key="load_mobile"):
                with st.spinner("Loading..."):
                    if st.session_state.data_handler.load_csv(uploaded_file):
                        st.session_state.data_loaded = True
                        st.rerun()
        
        # AI Status
        with st.expander("🤖 AI Status", expanded=False):
            if st.session_state.llm_handler.is_available():
                st.markdown("""
                <div class="status-success">
                    ✅ AI Model Ready
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-warning">
                    ⚠️ Basic mode (no AI)
                </div>
                """, unsafe_allow_html=True)

def render_quick_questions():
    """Render quick question buttons for mobile."""
    st.subheader("💡 Quick Questions")
    
    quick_questions = [
        "Projects in Region IV-B?",
        "Most expensive projects?",
        "Projects in Palawan?",
        "Top contractors?",
        "2024 completions?",
        "Drainage projects?"
    ]
    
    # Create responsive grid
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                # Expand the question for better context
                full_questions = [
                    "What flood control projects are in Region IV-B?",
                    "What are the most expensive flood control projects?",
                    "Show me all projects in Palawan province",
                    "Which contractor has completed the most projects?",
                    "Show me projects completed in 2024",
                    "Which projects involve drainage system construction?"
                ]
                
                # Add to query input
                st.session_state.quick_query = full_questions[i]
                st.rerun()

def main():
    """Main application function."""
    
    # Check if we're on mobile (simplified detection)
    is_mobile = st.session_state.get('is_mobile', False)
    
    # Header
    st.title("🌊 FloodWise PH")
    st.markdown("**Philippines Flood Control Intelligence Platform** - Ask questions about flood control projects, contractors, costs, locations, and mitigation works.")
    
    # Sidebar for data information and configuration
    render_mobile_sidebar()
    
    # Main chat interface
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="status-error">
            ❌ Dataset could not be loaded. Please check if the dataset file exists in the Dataset folder.
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Expected file: `Dataset/flood-control-projects-table_2025-09-20.csv`")
        
        # Show sample questions in mobile-friendly format
        with st.expander("💡 Sample Questions (Available when dataset is loaded)", expanded=True):
            sample_questions = [
                "What flood control projects are in Region IV-B?",
                "Show me all projects in Palawan province",
                "Which contractor has completed the most projects?",
                "What are the most expensive flood control projects?",
                "Show me projects completed in 2024",
                "What types of flood mitigation works are included?",
                "Which projects involve drainage system construction?",
                "Show me projects with contract costs over 50 million pesos",
                "What projects are implemented in Puerto Princesa City?",
                "Which district engineering offices have the most projects?"
            ]
            
            for question in sample_questions:
                st.write(f"• {question}")
        
        return
    
    # Quick questions for mobile users
    render_quick_questions()
    
    # Chat interface
    st.subheader("💬 Ask About Flood Control Projects")
    
    # Display chat history with mobile-friendly styling
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            render_chat_message("user", question, i)
            render_chat_message("assistant", answer, i)
            st.markdown("---")
    
    # Query input with mobile optimization
    with st.form("query_form", clear_on_submit=True):
        # Pre-fill with quick query if selected
        default_query = st.session_state.get('quick_query', '')
        if default_query:
            st.session_state.quick_query = ''  # Clear after use
        
        user_query = st.text_input(
            "Enter your question:",
            value=default_query,
            placeholder="e.g., What are the most expensive flood control projects in Manila?",
            help="Ask about project details, contractors, costs, completion dates, locations, or types of works"
        )
        
        # Mobile-friendly button layout
        col1, col2 = st.columns([2, 1])
        with col1:
            submit_button = st.form_submit_button("Ask Question", type="primary", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("Clear", use_container_width=True)
    
    # Handle form submissions
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit_button and user_query.strip():
        with st.spinner("🔍 Searching and generating response..."):
            # Search for relevant records
            relevant_records = st.session_state.data_handler.search_relevant_records(
                user_query, top_k=5
            )
            
            # Get context information
            context_info = st.session_state.data_handler.get_summary_stats()
            
            # Generate response
            response = st.session_state.llm_handler.generate_response(
                user_query, relevant_records, context_info
            )
            
            # Add to chat history
            st.session_state.chat_history.append((user_query, response))
            
            st.rerun()
    
    # Advanced features in mobile-friendly expander
    if st.session_state.data_loaded:
        with st.expander("🔍 Advanced Data Exploration"):
            st.subheader("Filter and Explore Data")
            
            # Get column information
            column_info = st.session_state.data_handler.get_column_info()
            
            # Create mobile-friendly filters
            filters = {}
            
            if column_info:
                selected_column = st.selectbox(
                    "Select column to filter by:", 
                    ["None"] + list(column_info.keys()),
                    help="Choose a column to filter the data"
                )
                
                if selected_column != "None":
                    filter_value = st.text_input(
                        f"Filter {selected_column} contains:",
                        help=f"Enter text to search within {selected_column}"
                    )
                    if filter_value:
                        filters[selected_column] = filter_value
                        
                        if st.button("Apply Filter", use_container_width=True):
                            filtered_df = st.session_state.data_handler.filter_records(filters)
                            st.success(f"Found {len(filtered_df)} matching records")
                            
                            if not filtered_df.empty:
                                # Mobile-friendly dataframe display
                                st.markdown("**Filtered Results (showing first 10 records):**")
                                
                                # Display key columns only for mobile
                                key_columns = ['ProjectDescription', 'Region', 'Province', 'Contractor', 'ContractCost']
                                display_columns = [col for col in key_columns if col in filtered_df.columns]
                                
                                if display_columns:
                                    st.dataframe(
                                        filtered_df[display_columns].head(10),
                                        use_container_width=True
                                    )
                                else:
                                    st.dataframe(filtered_df.head(10), use_container_width=True)

def show_footer():
    """Display mobile-friendly footer information."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><strong>FloodWise PH</strong><br>
            Philippines Flood Control Intelligence Platform<br>
            Built with Streamlit & OpenAI</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()