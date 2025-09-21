import logging
import os
import re
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the project modules
sys.path.append(str(Path(__file__).parent.absolute()))

import pandas as pd
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
    
    # Handle quick questions from sidebar
    if st.session_state.get('quick_question'):
        prompt = st.session_state.quick_question
        st.session_state.quick_question = None  # Clear it
        
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()
    
    # Handle filter suggestions from sidebar
    if st.session_state.get('filter_suggestion'):
        st.info(f"💡 **Suggested query:** {st.session_state.filter_suggestion}")
        if st.button("Use this query", key="use_filter_suggestion"):
            prompt = st.session_state.filter_suggestion
            st.session_state.filter_suggestion = None
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()
    
    # Show welcome message if no chat history
    if not st.session_state.get('chat_history', []):
        with st.chat_message("assistant"):
            st.markdown("""
            👋 **Welcome to FloodWise PH!** 
            
            I'm your AI assistant for exploring flood control projects across the Philippines. I have access to thousands of DPWH project records with detailed information about costs, contractors, locations, and project types.
            
            **🚀 Quick Start:**
            - Use the **Quick Filters** in the sidebar to narrow down your search
            - Try the **Quick Actions** buttons for instant insights
            - Ask me anything in natural language!
            
            **💡 Pro Tips:**
            - Be specific about locations: *"projects in Cebu City"*
            - Ask for comparisons: *"compare costs between regions"*
            - Request specific numbers: *"show me top 10 most expensive projects"*
            
            What would you like to explore? 🌊
            """)
    
    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    
    # Chat input with enhanced placeholder
    placeholder_text = "Ask about flood control projects... (e.g., 'Show me projects in Manila over 50M pesos')"
    if prompt := st.chat_input(placeholder_text):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing your query and searching through project data..."):
                try:
                    # Parse the query for number of results requested
                    num_match = re.search(r'(?:show|list|top|first|get)\s+(?:me\s+)?(\d+)(?:\s+results?)?', prompt, re.IGNORECASE)
                    num_results = int(num_match.group(1)) if num_match else 10  # Increased default
                    
                    # Limit to reasonable number
                    num_results = min(num_results, 50)
                    
                    # Search for relevant records
                    results = st.session_state.data_handler.search_relevant_records(prompt, top_k=num_results)
                    
                    # Generate response
                    if st.session_state.get('llm_handler') and st.session_state.llm_handler.is_available():
                        response = st.session_state.llm_handler.generate_response(
                            query=prompt,
                            results=results
                        )
                    else:
                        # Fallback response when AI is not available
                        if results:
                            response = f"**Found {len(results)} matching projects:**\n\n"
                            for i, project in enumerate(results[:5], 1):
                                response += f"**{i}. {project.get('ProjectDescription', 'N/A')[:100]}...**\n"
                                response += f"- **Location:** {project.get('Municipality', 'N/A')}, {project.get('Province', 'N/A')}\n"
                                response += f"- **Contractor:** {project.get('Contractor', 'N/A')}\n"
                                if project.get('ContractCost'):
                                    response += f"- **Cost:** ₱{project.get('ContractCost', 0):,.2f}\n"
                                response += "\n"
                            if len(results) > 5:
                                response += f"... and {len(results) - 5} more projects.\n"
                        else:
                            response = "No matching projects found. Try adjusting your search terms or using the filters in the sidebar."
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    # Show data source info
                    if results:
                        with st.expander(f"📊 View {len(results)} matching projects", expanded=False):
                            import pandas as pd
                            df_results = pd.DataFrame(results)
                            # Show key columns only
                            display_cols = ['ProjectDescription', 'Municipality', 'Province', 'Contractor', 'ContractCost']
                            display_cols = [col for col in display_cols if col in df_results.columns]
                            st.dataframe(df_results[display_cols], use_container_width=True)
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while processing your request: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    
                    # Show debug info in expander
                    with st.expander("🔧 Debug Information", expanded=False):
                        st.code(f"Error: {str(e)}")
                        st.write("Please try:")
                        st.write("- Rephrasing your question")
                        st.write("- Using the Quick Filters in the sidebar")
                        st.write("- Checking if the dataset is properly loaded")

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
    
    # Enhanced Sidebar
    with st.sidebar:
        # Header with logo and title
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #1f77b4; margin: 0;'>🌊 FloodWise PH</h1>
            <p style='color: #666; margin: 0; font-size: 0.9em;'>AI-Powered Flood Control Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset Status Section
        with st.expander("📊 Dataset Status", expanded=True):
            if st.session_state.get('data_loaded', False):
                try:
                    stats = st.session_state.data_handler.get_summary_stats()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Projects", f"{stats.get('total_records', 0):,}")
                    with col2:
                        st.metric("Regions", stats.get('unique_regions', 0))
                    
                    # Additional stats
                    if 'unique_provinces' in stats:
                        st.metric("Provinces", stats.get('unique_provinces', 0))
                    if 'date_range' in stats:
                        date_range = stats.get('date_range', {})
                        if date_range.get('min_year') and date_range.get('max_year'):
                            st.metric("Year Range", f"{date_range['min_year']} - {date_range['max_year']}")
                    
                    st.success("✅ Dataset Ready")
                except Exception as e:
                    st.success("✅ Dataset Loaded")
                    st.caption(f"Stats unavailable: {str(e)[:50]}...")
            else:
                st.warning("⏳ Loading dataset...")
                st.info("Please wait while we load the flood control projects data.")
        
        # Quick Filters Section
        if st.session_state.get('data_loaded', False):
            with st.expander("🔍 Quick Filters", expanded=False):
                st.markdown("**Filter your queries by:**")
                
                # Region filter
                try:
                    if hasattr(st.session_state.data_handler, 'df') and st.session_state.data_handler.df is not None:
                        regions = sorted(st.session_state.data_handler.df['Region'].dropna().unique())
                        selected_region = st.selectbox(
                            "Region", 
                            ["All Regions"] + list(regions),
                            key="region_filter"
                        )
                        
                        # Province filter (dependent on region)
                        if selected_region != "All Regions":
                            provinces = sorted(
                                st.session_state.data_handler.df[
                                    st.session_state.data_handler.df['Region'] == selected_region
                                ]['Province'].dropna().unique()
                            )
                            selected_province = st.selectbox(
                                "Province", 
                                ["All Provinces"] + list(provinces),
                                key="province_filter"
                            )
                        
                        # Year range filter
                        if 'CompletionYear' in st.session_state.data_handler.df.columns:
                            years = st.session_state.data_handler.df['CompletionYear'].dropna()
                            if len(years) > 0:
                                min_year = int(years.min())
                                max_year = int(years.max())
                                year_range = st.slider(
                                    "Completion Year Range",
                                    min_value=min_year,
                                    max_value=max_year,
                                    value=(min_year, max_year),
                                    key="year_filter"
                                )
                        
                        # Apply filters button
                        if st.button("🔍 Apply Filters", use_container_width=True):
                            filter_text = ""
                            if selected_region != "All Regions":
                                filter_text += f" in {selected_region}"
                                if 'selected_province' in locals() and selected_province != "All Provinces":
                                    filter_text += f", {selected_province}"
                            if 'year_range' in locals():
                                filter_text += f" completed between {year_range[0]} and {year_range[1]}"
                            
                            if filter_text:
                                st.session_state.filter_suggestion = f"Show me flood control projects{filter_text}"
                                st.success(f"Filter applied! Try: '{st.session_state.filter_suggestion}'")
                
                except Exception as e:
                    st.info("Filters will be available once data is fully loaded.")
        
        # AI Assistant Section
        with st.expander("🤖 AI Assistant", expanded=True):
            if st.session_state.get('llm_handler') and st.session_state.llm_handler.is_available():
                st.success("✅ AI Model Ready")
                st.caption("Using GPT-3.5-turbo for intelligent responses")
            else:
                st.warning("⚠️ Basic Mode Active")
                st.caption("Limited functionality without AI model")
            
            # Model info
            st.markdown("**Features:**")
            st.markdown("""
            - 🧠 Natural language understanding
            - 📊 Data analysis and insights
            - 🔍 Intelligent search and filtering
            - 📈 Cost and timeline analysis
            """)
        
        # Sample Questions Section
        with st.expander("💬 Sample Questions", expanded=False):
            st.markdown("**💰 Cost Analysis:**")
            st.markdown("""
            - *"What are the most expensive projects?"*
            - *"Show me projects over 100 million pesos"*
            - *"Compare costs by region"*
            """)
            
            st.markdown("**🏗️ Project Types:**")
            st.markdown("""
            - *"Tell me about drainage projects"*
            - *"Show me flood control structures"*
            - *"What types of projects are most common?"*
            """)
            
            st.markdown("**📍 Location-based:**")
            st.markdown("""
            - *"Projects in Metro Manila"*
            - *"Show me work in Cebu province"*
            - *"Which regions have the most projects?"*
            """)
            
            st.markdown("**👷 Contractors:**")
            st.markdown("""
            - *"Who are the top contractors?"*
            - *"Show me projects by [contractor name]"*
            - *"Compare contractor performance"*
            """)
            
            # Quick question buttons
            st.markdown("**Quick Actions:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💰 Top Expensive", use_container_width=True):
                    st.session_state.quick_question = "What are the 5 most expensive flood control projects?"
            with col2:
                if st.button("📊 Summary", use_container_width=True):
                    st.session_state.quick_question = "Give me a summary of the flood control projects dataset"
        
        # Data Insights Section
        if st.session_state.get('data_loaded', False):
            with st.expander("📈 Quick Insights", expanded=False):
                try:
                    df = st.session_state.data_handler.df
                    if df is not None:
                        # Top contractor
                        if 'Contractor' in df.columns:
                            top_contractor = df['Contractor'].value_counts().index[0]
                            contractor_count = df['Contractor'].value_counts().iloc[0]
                            st.metric("Top Contractor", top_contractor, f"{contractor_count} projects")
                        
                        # Most active region
                        if 'Region' in df.columns:
                            top_region = df['Region'].value_counts().index[0]
                            region_count = df['Region'].value_counts().iloc[0]
                            st.metric("Most Active Region", top_region, f"{region_count} projects")
                        
                        # Average cost (if available)
                        if 'ContractCost' in df.columns:
                            avg_cost = df['ContractCost'].mean()
                            if not pd.isna(avg_cost):
                                st.metric("Avg. Contract Cost", f"₱{avg_cost/1e6:.1f}M")
                
                except Exception as e:
                    st.info("Insights will be available once data is processed.")
        
        st.markdown("---")
        
        # Control Buttons Section
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                # Clear any cached data and reload
                if 'data_handler' in st.session_state:
                    st.session_state.data_loaded = False
                st.rerun()
        
        # Export Options
        if st.session_state.get('data_loaded', False):
            with st.expander("📥 Export Options", expanded=False):
                st.markdown("**Download Data:**")
                
                if st.button("📊 Export Chat History", use_container_width=True):
                    if st.session_state.get('chat_history'):
                        chat_text = "\n\n".join([
                            f"**{msg['role'].title()}:** {msg['content']}" 
                            for msg in st.session_state.chat_history
                        ])
                        st.download_button(
                            "💾 Download Chat",
                            chat_text,
                            file_name="floodwise_chat_history.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.info("No chat history to export")
                
                st.markdown("**Generate Reports:**")
                if st.button("📋 Generate Summary Report", use_container_width=True):
                    st.session_state.quick_question = "Generate a comprehensive summary report of all flood control projects including key statistics, top contractors, regional distribution, and cost analysis"
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            <p>🇵🇭 <strong>FloodWise PH</strong></p>
            <p>Powered by DPWH Data</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
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
