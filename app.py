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

# Enhanced UI with cohesive design system
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* CSS Variables for consistent theming */
    :root {
        --primary-blue: #0066CC;
        --primary-blue-light: #E6F3FF;
        --primary-blue-dark: #004499;
        --secondary-teal: #00A693;
        --accent-orange: #FF6B35;
        --text-primary: #1E293B;
        --text-secondary: #64748B;
        --text-muted: #94A3B8;
        --bg-primary: #FFFFFF;
        --bg-secondary: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-light: #E2E8F0;
        --border-medium: #CBD5E1;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
    }
    
    /* Base typography */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-secondary);
        color: var(--text-primary);
    }
    
    /* Remove Streamlit branding and optimize layout */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    header[data-testid="stHeader"] {
        height: 0;
        display: none;
    }
    
    .stApp > header {
        display: none;
    }
    
    /* Hide notifications */
    .stAlert[data-baseweb="notification"],
    div[data-testid="stNotificationContentSuccess"],
    div[data-testid="stNotificationContentInfo"],
    div[data-testid="stNotificationContentWarning"] {
        display: none !important;
    }
    
    /* Enhanced title styling */
    .main h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: var(--primary-blue);
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle styling */
    .main p {
        font-size: 1.1rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Section headers */
    .main h2, .main h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .main h2 {
        font-size: 1.5rem;
        border-bottom: 2px solid var(--primary-blue-light);
        padding-bottom: 0.5rem;
    }
    
    .main h3 {
        font-size: 1.25rem;
        color: var(--primary-blue);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        border-radius: var(--radius-md);
        border: 2px solid transparent;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%);
        color: white;
        min-height: 48px;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--primary-blue-dark) 0%, var(--primary-blue) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }
    
    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: var(--bg-card);
        color: var(--primary-blue);
        border: 2px solid var(--primary-blue);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--primary-blue-light);
        border-color: var(--primary-blue-dark);
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        border: none;
        background: transparent;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-teal) 100%);
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--primary-blue-light);
        color: var(--primary-blue);
    }
    
    /* Enhanced form styling */
    .stTextInput > div > div > input {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        padding: 0.875rem 1rem;
        border-radius: var(--radius-md);
        border: 2px solid var(--border-light);
        background: var(--bg-card);
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 3px var(--primary-blue-light);
        outline: none;
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 1rem;
        color: var(--primary-blue);
        box-shadow: var(--shadow-sm);
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-top: none;
        border-radius: 0 0 var(--radius-md) var(--radius-md);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* Card-like containers */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Status message styling */
    .status-error {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #FECACA;
        border-radius: var(--radius-md);
        padding: 1rem;
        color: #DC2626;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    
    .status-success {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #BBF7D0;
        border-radius: var(--radius-md);
        padding: 1rem;
        color: #16A34A;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    
    /* Chat message styling */
    .chat-message {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        font-family: 'Inter', sans-serif;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, var(--primary-blue-light) 0%, #F0F9FF 100%);
        border-color: var(--primary-blue);
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: var(--bg-card);
        margin-right: 2rem;
    }
    
    .chat-header {
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--primary-blue);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--bg-card);
        border-right: 1px solid var(--border-light);
    }
    
    /* Code blocks */
    .stCode {
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
    }
    
    /* Compact layout */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .main h1 {
            font-size: 2rem;
        }
        
        .main h2 {
            font-size: 1.25rem;
        }
        
        .stButton > button {
            font-size: 0.9rem;
            padding: 0.625rem 1rem;
            min-height: 44px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem 0.75rem;
        }
        
        .chat-message.user {
            margin-left: 0.5rem;
        }
        
        .chat-message.assistant {
            margin-right: 0.5rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #F1F5F9;
            --text-secondary: #CBD5E1;
            --text-muted: #64748B;
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-card: #334155;
            --border-light: #475569;
            --border-medium: #64748B;
        }
    }
    
    /* Smooth animations */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
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
    """Render individual chat message with clean formatting."""
    if message_type == "user":
        st.markdown("**🙋‍♂️ You:**")
        st.markdown(content)
    else:
        st.markdown("**🤖 Assistant:**")
        st.markdown(content)

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
    """Render quick question buttons for mobile - DISABLED per user request."""
    # Quick questions removed per user request
    pass

def main():
    """Main application function."""
    
    # Check if we're on mobile (simplified detection)
    is_mobile = st.session_state.get('is_mobile', False)
    
    # Enhanced Header with better typography and icons
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem;">🌊 FloodWise PH</h1>
        <p style="font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 0;">
            <strong>🇵🇭 Philippines Flood Control Intelligence Platform</strong>
        </p>
        <p style="color: var(--text-muted); margin-top: 0.5rem;">
            💡 Ask questions about projects • 🏗️ contractors • 💰 costs • 📍 locations • 🛡️ mitigation works
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Sample Questions Section (when data is loaded)
    if not st.session_state.chat_history:  # Only show when no chat history
        # Organize sample questions by category with better icons
        sample_categories = {
            "💰 Cost & Budget": [
                "💸 What are the most expensive projects in Cebu City?",
                "💵 What project has the least budget?",
                "📊 What's the average spending on flood control projects in Region VII?",
                "🔍 Show me projects under ₱10 million",
                "📈 Which projects went over budget?"
            ],
            "📍 Location-Based": [
                "🏝️ Show me all projects in Palawan",
                "🌏 What flood control projects are in Region IV-B?",
                "🏙️ How many projects are in Metro Manila?",
                "🗺️ Which province has the most projects?",
                "📍 List projects in Puerto Princesa City"
            ],
            "🏗️ Contractors": [
                "👷 Which contractor built the most projects?",
                "🏆 Who constructed the most expensive project?",
                "🔍 Show me projects by AZARRAGA CONSTRUCTION",
                "🌐 Which contractors work in multiple regions?",
                "📊 Top 5 contractors by project count"
            ],
            "📅 Timeline": [
                "🆕 What projects were completed in 2024?",
                "⏰ Show me recent flood control projects",
                "🔮 Which projects are scheduled for 2025?",
                "🚰 What drainage projects were finished in 2023?",
                "📜 Oldest projects in the database"
            ],
            "🔧 Project Types": [
                "🏗️ What types of flood control structures are there?",
                "🚰 Show me all drainage projects",
                "🌉 List bridge flood control projects",
                "🌊 What seawall projects exist?",
                "⛰️ Slope protection projects in mountainous areas"
            ],
            "📊 Analysis": [
                "📈 How many total projects are in the dataset?",
                "🗺️ What is the distribution of projects by region?",
                "📊 Show the trend of funding from 2020-2024",
                "📅 Which year had the most projects?",
                "⚖️ Compare Manila vs Cebu investments"
            ]
        }
        
        # Show total count
        total_samples = sum(len(questions) for questions in sample_categories.values())
        st.markdown(f"**{total_samples} sample questions organized by category:**")
        
        # Create tabs for different categories
        tabs = st.tabs(list(sample_categories.keys()))
        
        for i, (category, questions) in enumerate(sample_categories.items()):
            with tabs[i]:
                st.markdown(f"**Click any question to try it:**")
                
                # Create clickable buttons for each question
                for j, question in enumerate(questions):
                    if st.button(question, key=f"sample_{i}_{j}", use_container_width=True):
                        # Set the question in session state and rerun
                        st.session_state.quick_query = question
                        st.rerun()
                
                # Add a text note
                st.markdown("*💡 You can also type your own questions in the input box below.*")
    
    # Chat interface
    st.subheader("💬 Ask About Flood Control Projects")
    
    # Add sample questions button for users who already have chat history
    if st.session_state.chat_history:
        with st.expander("💡 Need inspiration? View sample questions", expanded=False):
            st.markdown("**🚀 Quick Examples:**")
            quick_examples = [
                "💰 What are the most expensive projects in Manila?",
                "🏗️ Which contractor has the most projects?",
                "🚰 Show me recent drainage projects",
                "📊 How many projects are in Region VII?",
                "💵 What's the average project cost in Palawan?"
            ]
            
            cols = st.columns(len(quick_examples))
            for i, example in enumerate(quick_examples):
                with cols[i]:
                    if st.button(example, key=f"quick_{i}", use_container_width=True):
                        st.session_state.quick_query = example
                        st.rerun()
    
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
            placeholder="e.g., What are the most expensive flood control projects in Cebu City?",
            help="💡 Ask about: 💰 Costs & budgets | 📍 Locations & regions | 🏗️ Contractors | 📅 Timelines | 🔧 Project types | 📊 Analysis & insights"
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
    
    # Advanced data exploration removed per user request

def show_footer():
    """Display enhanced footer with cohesive design."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: var(--text-muted); padding: 2rem 1rem; background: var(--bg-card); border-radius: var(--radius-lg); margin-top: 2rem; border: 1px solid var(--border-light);'>
            <h4 style='color: var(--primary-blue); margin-bottom: 0.5rem; font-weight: 600;'>🌊 FloodWise PH</h4>
            <p style='margin-bottom: 0.5rem; font-size: 1rem;'><strong>🇵🇭 Philippines Flood Control Intelligence Platform</strong></p>
            <p style='font-size: 0.9rem; margin-bottom: 1rem;'>Empowering informed decisions through intelligent data analysis</p>
            <div style='display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; font-size: 0.85rem;'>
                <span>🚀 Built with Streamlit</span>
                <span>🤖 Powered by AI</span>
                <span>📊 Real-time Analytics</span>
                <span>📱 Mobile Optimized</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()