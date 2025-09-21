import streamlit as st
import pandas as pd
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="FloodWise PH - Philippines Flood Control Intelligence",
    page_icon="🌊",
    layout="wide"
)

# Mobile-friendly styling
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            padding: 1rem;
        }
        [data-testid="stSidebar"] {
            width: 18rem !important;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
        @media (max-width: 768px) {
            [data-testid="stSidebar"] {
                display: none;
            }
            .block-container {
                padding: 1rem 0.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize handlers
data_handler = FloodControlDataHandler()
llm_handler = LLMHandler()

# Sidebar
st.sidebar.title("📊 FloodWise PH")
st.sidebar.markdown("Philippines Flood Control Intelligence")
st.sidebar.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Flood Control Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    data_handler.load_dataframe(df)
    st.sidebar.success("✅ Dataset loaded successfully")

    # Sidebar filters
    st.sidebar.subheader("🔍 Search / Filter")
    query = st.sidebar.text_input("Enter search keyword")
    top_k = st.sidebar.slider("Results to show", 1, 20, 5)
else:
    st.warning("Please upload a dataset to begin.")
    st.stop()

# Main content
st.title("🌊 FloodWise PH")
st.subheader("Flood Control Project Data Explorer")

# Show dataset preview
with st.expander("📂 Preview Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

# Query handler
if query:
    with st.spinner("Searching relevant data..."):
        results = data_handler.query(query, top_k=top_k)
        if not results.empty:
            st.success(f"Found {len(results)} matching records")
            st.dataframe(results, use_container_width=True)
        else:
            st.error("No matching records found.")

# LLM insights
st.markdown("### 🤖 AI Insights")
user_question = st.text_area("Ask a question about the dataset:")
if st.button("Generate Answer"):
    if user_question.strip():
        with st.spinner("Generating AI insights..."):
            answer = llm_handler.get_response(user_question, df)
            st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:14px;'>FloodWise PH © 2025 | Powered by Innovare & Cisco</p>",
    unsafe_allow_html=True
)
