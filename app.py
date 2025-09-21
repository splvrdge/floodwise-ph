import streamlit as st
import pandas as pd
import os
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="FloodWise PH - Philippines Flood Control Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling
st.markdown(
    """
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1200px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            width: 20rem !important;
        }
        
        /* Chat input styling */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            font-size: 16px;
            border-radius: 10px;
        }
        
        /* Metric styling */
        [data-testid="metric-container"] {
            background-color: #f0f2f6;
            border: 1px solid #e0e0e0;
            padding: 0.5rem;
            border-radius: 10px;
            margin: 0.25rem 0;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 10px;
            border: none;
            background: linear-gradient(90deg, #1f77b4, #17a2b8);
            color: white;
            font-weight: 600;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 10px;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            [data-testid="stSidebar"] {
                width: 16rem !important;
            }
            .main .block-container {
                padding: 0.5rem;
            }
        }
        
        /* Custom header styling */
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #17a2b8);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Status indicators */
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.25rem 0;
        }
        
        .status-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.25rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize handlers
@st.cache_resource
def initialize_handlers():
    """Initialize handlers."""
    try:
        data_handler = FloodControlDataHandler()
        llm_handler = LLMHandler()
        return data_handler, llm_handler, None
    except Exception as e:
        return None, None, str(e)

data_handler, llm_handler, init_error = initialize_handlers()

if init_error:
    st.error(f"Failed to initialize handlers: {init_error}")
    st.stop()

# Load dataset automatically
@st.cache_data
def load_dataset():
    """Load the dataset with caching."""
    dataset_path = "Dataset/flood-control-projects-table_2025-09-20.csv"
    
    if os.path.exists(dataset_path):
        success = data_handler.load_csv_from_path(dataset_path)
        if success:
            return True, data_handler.df, None
        else:
            return False, None, "Failed to load dataset"
    else:
        return False, None, f"Dataset file not found: {dataset_path}"

# Load dataset on startup
if not st.session_state.data_loaded:
    with st.spinner("🔄 Loading DPWH flood control projects dataset..."):
        success, df, error = load_dataset()
        
    if success:
        st.session_state.data_loaded = True
        st.session_state.df = df
    else:
        st.error(f"❌ {error}")
        st.stop()
else:
    df = st.session_state.df

# Custom header
st.markdown(
    """
    <div class="main-header">
        <h1>🌊 FloodWise PH</h1>
        <p>AI-Powered Philippines Flood Control Intelligence Platform</p>
        <p>📊 Analyzing DPWH Infrastructure Projects with Advanced AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Dataset Overview")
    
    # Dataset statistics
    if st.session_state.data_loaded:
        try:
            stats = data_handler.get_summary_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Projects", f"{stats.get('total_records', len(df)):,}")
            with col2:
                st.metric("Regions", stats.get('unique_regions', df['Region'].nunique() if 'Region' in df.columns else 0))
            
            if 'Province' in df.columns:
                st.metric("Provinces", df['Province'].nunique())
            
            if 'ContractCost' in df.columns:
                avg_cost = df['ContractCost'].mean()
                if not pd.isna(avg_cost):
                    st.metric("Avg Cost", f"₱{avg_cost/1e6:.1f}M")
                    
        except Exception as e:
            st.metric("Total Projects", f"{len(df):,}")
    
    st.markdown("---")
    
    # AI Status
    st.markdown("### 🤖 AI Assistant Status")
    if llm_handler and llm_handler.is_available():
        st.success("✅ TinyLlama API Ready")
        st.caption("🌐 Hugging Face API")
    else:
        st.warning("⚠️ API Not Available")
        st.caption("Using fallback responses")
    
    st.markdown("---")
    
    # Quick filters
    st.markdown("### 🔍 Quick Filters")
    
    # Region filter
    if 'Region' in df.columns:
        regions = ['All Regions'] + sorted(df['Region'].dropna().unique().tolist())
        selected_region = st.selectbox("Region", regions)
        
        # Province filter
        if selected_region != 'All Regions':
            provinces = ['All Provinces'] + sorted(
                df[df['Region'] == selected_region]['Province'].dropna().unique().tolist()
            )
            selected_province = st.selectbox("Province", provinces)
    
    # Cost range filter
    if 'ContractCost' in df.columns:
        cost_data = df['ContractCost'].dropna()
        if len(cost_data) > 0:
            min_cost = float(cost_data.min())
            max_cost = float(cost_data.max())
            cost_range = st.slider(
                "Cost Range (₱M)",
                min_value=min_cost/1e6,
                max_value=max_cost/1e6,
                value=(min_cost/1e6, max_cost/1e6),
                format="%.1f"
            )
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💰 Top Expensive", use_container_width=True):
            st.session_state.quick_query = "Show me the 10 most expensive flood control projects"
    with col2:
        if st.button("📊 Summary", use_container_width=True):
            st.session_state.quick_query = "Give me a summary of the flood control projects"
    
    if st.button("🏗️ By Contractor", use_container_width=True):
        st.session_state.quick_query = "Which contractors have the most projects?"
    
    if st.button("📍 By Region", use_container_width=True):
        st.session_state.quick_query = "Show me project distribution by region"

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat interface
    st.markdown("### 💬 Ask FloodWise AI")
    
    # Show welcome message if no chat history
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown("""
            👋 **Welcome to FloodWise PH!** 
            
            I can help you analyze flood control projects across the Philippines using AI.
            
            **💡 Try asking:**
            - *"What are the most expensive projects?"*
            - *"Show me projects in Metro Manila"*
            - *"Which contractor has the most projects?"*
            
            🤖 **Powered by TinyLlama AI via Hugging Face**
            """)
    
    # Handle quick queries
    if 'quick_query' in st.session_state:
        user_input = st.session_state.quick_query
        del st.session_state.quick_query
    else:
        user_input = st.chat_input("Ask about flood control projects... (e.g., 'Show projects in Manila over 50M pesos')")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Process user input
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing your query..."):
                try:
                    # Search for relevant data
                    results = data_handler.search_relevant_records(user_input, top_k=15)
                    
                    # Generate response
                    response = llm_handler.generate_response(user_input, results)
                    
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Show data source
                    if results:
                        with st.expander(f"📊 View {len(results)} matching projects"):
                            results_df = pd.DataFrame(results)
                            display_cols = ['ProjectDescription', 'Municipality', 'Province', 'Contractor', 'ContractCost']
                            display_cols = [col for col in display_cols if col in results_df.columns]
                            st.dataframe(results_df[display_cols], use_container_width=True)
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

with col2:
    # Dataset overview
    st.markdown("### 📈 Dataset Insights")
    
    with st.expander("📊 Quick Stats", expanded=True):
        if 'Region' in df.columns:
            st.markdown("**Top 5 Regions by Projects:**")
            top_regions = df['Region'].value_counts().head(5)
            for region, count in top_regions.items():
                st.write(f"• {region}: {count:,}")
        
        if 'Contractor' in df.columns:
            st.markdown("**Top 5 Contractors:**")
            top_contractors = df['Contractor'].value_counts().head(5)
            for contractor, count in top_contractors.items():
                st.write(f"• {contractor}: {count:,}")
    
    with st.expander("🔍 Search Data", expanded=False):
        search_query = st.text_input("Search projects:")
        search_limit = st.slider("Results", 1, 50, 10)
        
        if st.button("🔍 Search", use_container_width=True) and search_query:
            try:
                search_results = data_handler.search_relevant_records(search_query, top_k=search_limit)
                if search_results:
                    st.success(f"Found {len(search_results)} results")
                    results_df = pd.DataFrame(search_results)
                    st.dataframe(results_df[['ProjectDescription', 'Municipality', 'ContractCost']], use_container_width=True)
                else:
                    st.warning("No results found")
            except Exception as e:
                st.error(f"Search error: {e}")
    
    # Control buttons
    st.markdown("### ⚙️ Controls")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("📥 Export Chat", use_container_width=True):
        if st.session_state.chat_history:
            chat_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in st.session_state.chat_history
            ])
            st.download_button(
                "💾 Download",
                chat_text,
                file_name="floodwise_chat.txt",
                mime="text/plain",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <strong>🌊 FloodWise PH</strong> • 🇵🇭 Philippines Flood Control Intelligence<br>
        <small>Powered by DPWH Data • Built with Streamlit & TinyLlama AI</small>
    </div>
    """,
    unsafe_allow_html=True
)
