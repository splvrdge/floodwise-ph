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
        # Try to get API key from Streamlit secrets first, then environment variables
        api_key = None
        
        # Check Streamlit secrets (for Streamlit Cloud deployment)
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            # Fall back to environment variables (for local development)
            api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            try:
                self.client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                st.error(f"⚠️ Failed to initialize OpenAI client: {str(e)}")
                st.info("The app will continue to work with basic functionality (no AI responses).")
                self.client = None
        else:
            st.warning("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in Streamlit secrets or .env file.")
    
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
