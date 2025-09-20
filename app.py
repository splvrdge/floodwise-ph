import streamlit as st
import pandas as pd
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler
import os
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="FloodWise PH - Philippines Flood Control Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def main():
    """Main application function."""
    
    # Header
    st.title("🌊 FloodWise PH")
    st.markdown("**Philippines Flood Control Intelligence Platform** - Ask questions about flood control projects, contractors, costs, locations, and mitigation works.")
    
    # Sidebar for data information and configuration
    with st.sidebar:
        # Data information
        if st.session_state.data_loaded:
            st.header("📊 Dataset Information")
            st.success("✅ DPWH Flood Control Projects Dataset Loaded")
            
            # Display dataset summary
            stats = st.session_state.data_handler.get_summary_stats()
            st.write(f"**Total Records:** {stats['total_records']:,}")
            st.write(f"**Columns:** {len(stats['columns'])}")
            st.write(f"**Coverage:** All Philippine regions")
            st.write(f"**Time Period:** 2021-2024")
            
            with st.expander("📋 View Column Details"):
                for col in stats['columns']:
                    st.write(f"• {col}")
        else:
            st.header("📁 Dataset Status")
            st.warning("⚠️ Dataset not found or failed to load")
            
            # Optional manual upload as fallback
            st.subheader("Manual Upload (Fallback)")
            uploaded_file = st.file_uploader(
                "Upload CSV file with flood control project data",
                type=['csv'],
                help="Upload a CSV file containing flood control project information"
            )
            
            if uploaded_file is not None:
                if st.button("Load Uploaded Data"):
                    with st.spinner("Loading and processing data..."):
                        if st.session_state.data_handler.load_csv(uploaded_file):
                            st.session_state.data_loaded = True
                            st.rerun()
        
        # LLM Configuration
        st.header("🤖 AI Configuration")
        model_info = st.session_state.llm_handler.get_model_info()
        
        if st.session_state.llm_handler.is_available():
            st.success("✅ AI Model Ready")
            st.write(f"**Model:** {model_info['model']}")
        else:
            st.warning("⚠️ AI Model Not Configured")
            st.info("Add your OpenAI API key to .env file for enhanced responses")
    
    # Main chat interface
    if not st.session_state.data_loaded:
        st.error("❌ Dataset could not be loaded. Please check if the dataset file exists in the Dataset folder.")
        st.info("💡 Expected file: `Dataset/flood-control-projects-table_2025-09-20.csv`")
        
        # Show sample questions for when data is available
        st.subheader("💡 Sample Questions (Available when dataset is loaded):")
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
    
    # Chat interface
    st.subheader("💬 Ask About Flood Control Projects")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.write(f"**You:** {question}")
            st.write(f"**Assistant:** {answer}")
            st.divider()
    
    # Query input
    with st.form("query_form", clear_on_submit=True):
        user_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the most expensive flood control projects in Manila?",
            help="Ask about project details, contractors, costs, completion dates, locations, or types of works"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Ask Question", type="primary")
        with col2:
            clear_button = st.form_submit_button("Clear History")
    
    # Handle form submissions
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit_button and user_query.strip():
        with st.spinner("Searching and generating response..."):
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
    
    # Advanced features in expander
    if st.session_state.data_loaded:
        with st.expander("🔍 Advanced Data Exploration"):
            st.subheader("Filter and Explore Data")
            
            # Get column information
            column_info = st.session_state.data_handler.get_column_info()
            
            # Create filters
            filters = {}
            col1, col2 = st.columns(2)
            
            with col1:
                if column_info:
                    selected_column = st.selectbox("Select column to filter by:", 
                                                 ["None"] + list(column_info.keys()))
                    
                    if selected_column != "None":
                        filter_value = st.text_input(f"Filter {selected_column} contains:")
                        if filter_value:
                            filters[selected_column] = filter_value
            
            with col2:
                if filters:
                    if st.button("Apply Filter"):
                        filtered_df = st.session_state.data_handler.filter_records(filters)
                        st.write(f"Found {len(filtered_df)} matching records")
                        if not filtered_df.empty:
                            st.dataframe(filtered_df.head(10))

def show_footer():
    """Display footer information."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>FloodWise PH - Philippines Flood Control Intelligence Platform | Built with Streamlit & OpenAI</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()
