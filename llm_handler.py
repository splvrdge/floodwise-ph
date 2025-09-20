import openai
import os
from typing import List, Dict, Any, Optional
import streamlit as st
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class LLMHandler:
    """Handles LLM interactions for generating responses about flood control projects."""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-3.5-turbo"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        api_key = None
        
        # Debug: Check what's available in secrets
        try:
            # Try to access secrets
            if hasattr(st, 'secrets') and st.secrets:
                # Check if OPENAI_API_KEY exists in secrets
                if "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
                    st.success("✅ API key found in Streamlit secrets")
                else:
                    st.warning("⚠️ OPENAI_API_KEY not found in Streamlit secrets")
                    # Show available keys for debugging
                    available_keys = list(st.secrets.keys()) if st.secrets else []
                    if available_keys:
                        st.info(f"Available secret keys: {available_keys}")
            else:
                st.warning("⚠️ Streamlit secrets not available")
        except Exception as e:
            st.error(f"Error accessing Streamlit secrets: {str(e)}")
        
        # Fallback to environment variables if no API key found
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                st.info("✅ API key found in environment variables")
        
        if api_key:
            # Try multiple initialization strategies to handle httpx compatibility
            strategies = [
                ("Custom HTTP Client", self._init_with_custom_http_client),
                ("Default Client", self._init_with_default_client),
                ("Legacy Client", self._init_with_legacy_approach)
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    self.client = strategy_func(api_key)
                    if self.client:
                        st.success(f"✅ OpenAI client initialized successfully using {strategy_name}")
                        return
                except Exception as e:
                    st.warning(f"❌ {strategy_name} failed: {str(e)}")
                    continue
            
            # If all strategies fail
            st.error("⚠️ All OpenAI initialization strategies failed")
            st.info("The app will continue with basic functionality (no AI responses)")
            self.client = None
        else:
            st.error("⚠️ No OpenAI API key found in secrets or environment")
    
    def _init_with_custom_http_client(self, api_key: str):
        """Initialize with custom HTTP client to avoid proxies issue."""
        import httpx
        
        # Create custom HTTP client without problematic parameters
        http_client = httpx.Client(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True
        )
        
        return openai.OpenAI(
            api_key=api_key,
            http_client=http_client
        )
    
    def _init_with_default_client(self, api_key: str):
        """Initialize with default OpenAI client."""
        return openai.OpenAI(api_key=api_key)
    
    def _init_with_legacy_approach(self, api_key: str):
        """Initialize using legacy OpenAI approach."""
        # Set API key globally for legacy compatibility
        openai.api_key = api_key
        
        # Create a wrapper that mimics the new client interface
        class LegacyWrapper:
            def __init__(self):
                self.chat = self
                self.completions = self
            
            def create(self, **kwargs):
                return openai.ChatCompletion.create(**kwargs)
        
        return LegacyWrapper()
    
    def generate_response(self, query: str, relevant_records: List[Dict[str, Any]], 
                         context_info: Dict[str, Any] = None) -> str:
        """Generate a response based on the query and relevant flood control project data."""
        
        if not self.client:
            return self._fallback_response(query, relevant_records)
        
        try:
            # Prepare context from relevant records
            context = self._prepare_context(relevant_records, context_info)
            
            # Create the prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(query, context)
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"Error generating LLM response: {str(e)}")
            return self._fallback_response(query, relevant_records)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return """You are a knowledgeable assistant specializing in flood control projects in the Philippines. 
        You have access to a database of flood control project information and should provide accurate, 
        helpful responses based on the provided data.

        Guidelines:
        1. Answer questions directly and concisely
        2. Use specific data from the provided records when available
        3. If asked about costs, dates, locations, contractors, or project types, reference the exact data
        4. If the data doesn't contain information to answer the question, clearly state this
        5. Focus on flood mitigation, drainage, and water management projects
        6. Do not generate graphs, plots, or visual content - text responses only
        7. Be helpful and informative while staying within the scope of the provided data"""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create the user prompt with query and context."""
        return f"""Question: {query}

Available flood control project data:
{context}

Please provide a comprehensive answer based on the available data. If the data doesn't contain 
sufficient information to fully answer the question, please indicate what information is missing."""
    
    def _prepare_context(self, relevant_records: List[Dict[str, Any]], 
                        context_info: Dict[str, Any] = None) -> str:
        """Prepare context string from relevant records."""
        if not relevant_records:
            return "No relevant flood control project records found for this query."
        
        context_parts = []
        
        # Add dataset overview if available
        if context_info:
            context_parts.append(f"Dataset contains {context_info.get('total_records', 'unknown')} total records.")
            context_parts.append(f"Available columns: {', '.join(context_info.get('columns', []))}")
            context_parts.append("")
        
        # Add relevant records
        context_parts.append("Relevant project records:")
        
        for i, record in enumerate(relevant_records[:5], 1):  # Limit to top 5 records
            context_parts.append(f"\nRecord {i}:")
            for key, value in record.items():
                if key != 'similarity_score' and value is not None and str(value).strip():
                    context_parts.append(f"  {key}: {value}")
        
        return "\n".join(context_parts)
    
    def _fallback_response(self, query: str, relevant_records: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when LLM is not available."""
        if not relevant_records:
            return """I couldn't find any relevant flood control project records for your query. 
            Please try rephrasing your question or check if the CSV data contains the information you're looking for."""
        
        # Create a simple response based on available data
        response_parts = [
            f"Based on the available flood control project data, I found {len(relevant_records)} relevant record(s):",
            ""
        ]
        
        for i, record in enumerate(relevant_records[:3], 1):
            response_parts.append(f"Project {i}:")
            for key, value in record.items():
                if key != 'similarity_score' and value is not None and str(value).strip():
                    response_parts.append(f"  • {key}: {value}")
            response_parts.append("")
        
        response_parts.append("Note: This is a basic response. For more detailed analysis, please configure an OpenAI API key.")
        
        return "\n".join(response_parts)
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "provider": "OpenAI",
            "available": str(self.is_available())
        }
