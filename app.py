import streamlit as st
import os
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler
from mobile_utils import optimize_for_mobile

# Page configuration
st.set_page_config(
    page_title="FloodWise PH - Philippines Flood Control Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="auto"
)

# Apply mobile optimizations
optimize_for_mobile()

# Global styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --blue: #0066CC; --blue-dark: #004499; --teal: #00A693;
        --text: #1E293B; --text-light: #64748B; --bg: #F8FAFC; --card: #FFFFFF;
        --border: #E2E8F0; --shadow: 0 1px 3px rgba(0,0,0,.1);
    }
    body, .stApp { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); }
    h1,h2,h3 { font-weight: 600; }
    .status { border-radius:8px;padding:1rem;margin:.5rem 0;box-shadow:var(--shadow); }
    .status.success { background:#F0FDF4; border:1px solid #BBF7D0; color:#16A34A; }
    .status.error { background:#FEF2F2; border:1px solid #FECACA; color:#DC2626; }
    .status.warn { background:#FFFBEB; border:1px solid #FDE68A; color:#B45309; }
    .chat { padding:1rem;border-radius:10px;margin:0.5rem 0;box-shadow:var(--shadow); }
    .chat.user { background:#E6F3FF; }
    .chat.assistant { background:#fff; }
    footer { text-align:center;color:var(--text-light);font-size:0.85rem;margin-top:2rem; }
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

# Try loading dataset automatically
if not st.session_state.data_loaded:
    dataset_paths = [
        "Dataset/flood-control-projects-table_2025-09-20.csv",
        "./Dataset/flood-control-projects-table_2025-09-20.csv",
        os.path.join(os.path.dirname(__file__), "Dataset", "flood-control-projects-table_2025-09-20.csv"),
    ]
    for path in dataset_paths:
        if os.path.exists(path) and st.session_state.data_handler.load_csv_from_path(path):
            st.session_state.data_loaded = True
            break

# Components
def sidebar():
    with st.sidebar:
        st.markdown("### 📊 Dataset")
        if st.session_state.data_loaded:
            stats = st.session_state.data_handler.get_summary_stats()
            st.markdown(
                f"<div class='status success'>✅ Loaded {stats['total_records']:,} records<br>"
                f"📋 {len(stats['columns'])} columns<br>"
                f"🌏 {stats['unique_regions']} regions<br>"
                f"📅 {stats['date_range']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='status warn'>⚠️ Dataset not loaded</div>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded and st.button("Load"):
                if st.session_state.data_handler.load_csv(uploaded):
                    st.session_state.data_loaded = True
                    st.rerun()

        st.markdown("### 🤖 AI")
        if st.session_state.llm_handler.is_available():
            st.markdown("<div class='status success'>✅ AI Model Ready</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status warn'>⚠️ Basic mode (no AI)</div>", unsafe_allow_html=True)

def chat_ui():
    st.subheader("💬 Ask About Flood Control Projects")
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"<div class='chat user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat assistant'><b>Assistant:</b> {a}</div>", unsafe_allow_html=True)

    with st.form("query_form", clear_on_submit=True):
        query = st.text_input("Enter your question:")
        col1, col2 = st.columns([2, 1])
        with col1:
            ask = st.form_submit_button("Ask", type="primary", use_container_width=True)
        with col2:
            clear = st.form_submit_button("Clear", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()
    if ask and query.strip():
        with st.spinner("Processing..."):
            records = st.session_state.data_handler.search_relevant_records(query, top_k=5)
            context = st.session_state.data_handler.get_summary_stats()
            answer = st.session_state.llm_handler.generate_response(query, records, context)
            st.session_state.chat_history.append((query, answer))
            st.rerun()

def footer():
    st.markdown(
        "<footer>🌊 FloodWise PH • 🇵🇭 Philippines Flood Control Intelligence<br>"
        "🚀 Built with Streamlit • 🤖 AI-powered • 📱 Mobile Ready</footer>",
        unsafe_allow_html=True,
    )

# Main
def main():
    st.markdown("<h1 style='text-align:center'>🌊 FloodWise PH</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:var(--text-light)'>Philippines Flood Control Intelligence</p>", unsafe_allow_html=True)

    sidebar()
    if st.session_state.data_loaded:
        chat_ui()
    else:
        st.error("❌ Dataset could not be loaded. Please upload manually.")

    footer()

if __name__ == "__main__":
    main()
