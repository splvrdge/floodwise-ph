import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        if not self.api_key:
            logger.error("HUGGINGFACE_API_KEY not found in environment variables")
        
    def is_available(self):
        """Check if the API is available"""
        return True  # Always return True since we're using API
    
    def generate_response(self, query, results):
        """Generate a response using TinyLlama API"""
        try:
            # Prepare the data summary
            if not results:
                return "No matching flood control projects found for your query."
            
            # Create a simple summary of the data
            data_summary = f"Found {len(results)} flood control projects:\n\n"
            for i, project in enumerate(results[:5], 1):
                desc = project.get('ProjectDescription', 'Unknown project')[:80]
                location = f"{project.get('Municipality', 'Unknown')}, {project.get('Province', 'Unknown')}"
                cost = project.get('ContractCost', 0)
                contractor = project.get('Contractor', 'Unknown')
                
                data_summary += f"{i}. {desc}\n"
                data_summary += f"   Location: {location}\n"
                data_summary += f"   Contractor: {contractor}\n"
                if cost and cost > 0:
                    data_summary += f"   Cost: ₱{cost:,.2f}\n"
                data_summary += "\n"
            
            if len(results) > 5:
                data_summary += f"...and {len(results) - 5} more projects."
            
            # Simple prompt for TinyLlama
            prompt = f"""You are analyzing Philippine flood control projects. 

User question: {query}

Project data:
{data_summary}

Please provide a helpful analysis of this data to answer the user's question. Be specific and use the actual data provided."""

            # Call the API
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').replace(prompt, '').strip()
            
            # Fallback if API fails
            return self._simple_fallback(query, results)
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return self._simple_fallback(query, results)
    
    def _simple_fallback(self, query, results):
        """Simple fallback response"""
        if not results:
            return "No projects found matching your query."
        
        response = f"Found {len(results)} flood control projects:\n\n"
        
        for i, project in enumerate(results[:3], 1):
            desc = project.get('ProjectDescription', 'Unknown project')[:60]
            location = f"{project.get('Municipality', '')}, {project.get('Province', '')}"
            cost = project.get('ContractCost', 0)
            
            response += f"**{i}. {desc}**\n"
            response += f"📍 {location}\n"
            if cost and cost > 0:
                response += f"💰 ₱{cost:,.2f}\n"
            response += "\n"
        
        if len(results) > 3:
            response += f"...and {len(results) - 3} more projects."
        
        return response
