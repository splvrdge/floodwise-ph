import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import streamlit as st
from enum import Enum
from dotenv import load_dotenv

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for Hugging Face availability
HUGGINGFACE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
    logger.info("Hugging Face transformers available")
except ImportError:
    logger.warning("Hugging Face transformers not available. App will run in fallback mode.")
    # Don't raise error - allow app to continue without TinyLlama
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    torch = None

class QueryType(Enum):
    """Types of user queries that can be handled."""
    GENERAL_CONVERSATION = "general_conversation"
    FLOOD_CONTROL_QUERY = "flood_control_query"
    DATA_ANALYSIS = "data_analysis"
    UNKNOWN = "unknown"

class LLMHandler:
    """Handles LLM interactions for generating responses about flood control projects.
    
    This class provides an interface for both OpenAI and Hugging Face models.
    """
    
    # Common greeting phrases that indicate a general conversation
    GENERAL_PHRASES = [
        "hi", "hello", "hey", "how are you", "what's up", "good morning", 
        "good afternoon", "good evening", "hi there", "hey there", "greetings"
    ]
    
    # Keywords that indicate a flood control-related query
    FLOOD_KEYWORDS = [
        "flood", "drainage", "control", "project", "contract", "cost",
        "location", "region", "province", "municipality", "contractor",
        "flood control", "flood mitigation", "drainage system"
    ]
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the LLM handler with TinyLlama model.
        
        Args:
            model_name: Name of the Hugging Face model to use. 
                       Default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                       Other options: "microsoft/DialoGPT-medium", "microsoft/DialoGPT-small"
        """
        self.model = model_name
        self.tokenizer = None
        self.hf_model = None
        self.pipeline = None
        self.available = False
        self.model_type = "huggingface"  # Only support Hugging Face models
        
        # Initialize TinyLlama model
        self._initialize_huggingface_model()
    
    
    def _initialize_huggingface_model(self):
        """Initialize TinyLlama model with cloud-friendly error handling."""
        if not HUGGINGFACE_AVAILABLE:
            logger.warning("Hugging Face transformers not available. Please install with: pip install transformers torch")
            self.available = False
            return
            
        try:
            logger.info(f"Loading TinyLlama model: {self.model}")
            
            # Check if running on Streamlit Cloud (limited resources)
            import os
            is_cloud = any(key in os.environ for key in [
                'STREAMLIT_SERVER_RUNNING_REMOTELY', 
                'STREAMLIT_CLOUD',
                'STREAMLIT_SERVER_RUN_ON_UPDATE',
                'STREAMLIT_SHARING',
                'STREAMLIT_CLOUD_ENVIRONMENT'
            ]) or '/mount/src/' in os.getcwd()
            
            if is_cloud:
                logger.warning("Detected cloud environment - TinyLlama may not work due to memory constraints")
                # Don't try to load the model on cloud, just set as unavailable
                self.available = False
                logger.info("Skipping TinyLlama loading on cloud - using fallback mode")
                return
            
            # Determine device and dtype with better compatibility
            if torch and torch.cuda.is_available():
                device = 0  # Use first GPU
                model_dtype = torch.float16  # Use float16 for better compatibility
                logger.info("Using CUDA GPU")
            else:
                device = -1  # Use CPU
                model_dtype = torch.float32 if torch else None
                logger.info("Using CPU")
            
            # Create pipeline with explicit device handling (avoid device_map="auto")
            try:
                logger.info("Attempting to load TinyLlama with pipeline approach...")
                
                # Use the new 'dtype' parameter instead of deprecated 'torch_dtype'
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    dtype=model_dtype,
                    device=device,
                    trust_remote_code=True,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                    }
                )
                logger.info(f"Successfully loaded {self.model} with pipeline approach")
            except Exception as pipeline_error:
                logger.warning(f"Pipeline approach failed: {pipeline_error}")
                logger.info("Trying manual model loading...")
                
                # Fallback: Manual model and tokenizer loading
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model manually with proper device handling
                model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                # Move to device after loading
                if torch and torch.cuda.is_available():
                    model = model.to('cuda')
                
                # Create pipeline with loaded model
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=device
                )
                logger.info(f"Successfully loaded {self.model} with manual approach")
            
            # Store tokenizer reference for chat template
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                self.tokenizer = self.pipeline.tokenizer
            
            logger.info(f"TinyLlama model ready with {model_dtype} precision")
            self.available = True
            
        except Exception as e:
            logger.error(f"Failed to initialize TinyLlama model: {e}")
            self.tokenizer = None
            self.pipeline = None
            self.available = False
            # Don't show error in Streamlit UI to avoid breaking the app
            logger.warning("TinyLlama unavailable - app will run in fallback mode")
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of query to determine response format."""
        query = query.lower()
        
        if any(phrase in query for phrase in ["most expensive", "highest cost", "largest project"]):
            return "max_cost"
        elif any(phrase in query for phrase in ["most projects", "most active", "most frequent", "prominent contractor"]):
            return "contractor_frequency"
        elif any(phrase in query for phrase in ["location", "region", "province", "municipality"]):
            return "location"
        return "general"
    
    def _get_system_prompt(self, query_type: str) -> str:
        """Get the system prompt for TinyLlama to handle all flood control queries."""
        return """You are FloodWise PH, an expert AI assistant specializing in Philippine flood control infrastructure. You analyze DPWH (Department of Public Works and Highways) project data to provide comprehensive insights.

🌊 YOUR EXPERTISE:
- Philippine flood control projects and infrastructure
- DPWH project data analysis and interpretation
- Cost analysis, contractor performance, and regional patterns
- Project types: drainage systems, flood walls, river improvements, etc.

📊 RESPONSE GUIDELINES:
1. ALWAYS base responses ONLY on the provided project data
2. Use clear, professional language for government officials and citizens
3. Provide specific numbers, costs, locations, and dates from the data
4. Format monetary values with ₱ symbol and commas (₱1,234,567.89)
5. Use emojis strategically for visual appeal: 🌊💰📍📊🏗️📅⚡🏆
6. If data is missing, explicitly state "Data not available"
7. Explain the significance of findings for flood control planning

🎯 RESPONSE STRUCTURE:
- Direct answer to the user's question
- Key insights (2-3 most important findings)
- Supporting data with specific details
- Context about why this matters for flood control

📋 DATA FIELDS YOU'LL ANALYZE:
- ProjectDescription: What the project does
- Municipality, Province, Region: Where it's located
- ContractCost, ABC: How much it costs
- Contractor: Who's building it
- CompletionYear: When it was/will be finished
- TypeofWork: What kind of flood control work

Always provide helpful, accurate, and actionable insights based on the flood control project data provided to you."""
    
    def _prepare_context(self, relevant_records: List[Dict[str, Any]], 
                        context_info: Dict[str, Any] = None) -> str:
        """Prepare project data for TinyLlama to analyze."""
        if not relevant_records:
            return "No relevant flood control project records found for this query."
        
        context_parts = []
        
        # Add analysis scope
        total_records = len(relevant_records)
        context_parts.append(f"📊 FLOOD CONTROL PROJECT DATA ({total_records} projects found):")
        context_parts.append("")
        
        # Present raw data for TinyLlama to analyze
        for i, record in enumerate(relevant_records[:15], 1):  # Show more projects for better analysis
            context_parts.append(f"PROJECT {i}:")
            
            # All available fields - let TinyLlama decide what's important
            for key, value in record.items():
                if value is not None and str(value).strip() and str(value).lower() not in ['n/a', 'none', 'null', '']:
                    # Clean up the value
                    if isinstance(value, str):
                        value = value.strip()
                    context_parts.append(f"  {key}: {value}")
            
            context_parts.append("")  # Add spacing between records
            
        if len(relevant_records) > 15:
            context_parts.append(f"... and {len(relevant_records) - 15} more projects in the dataset.")
            
        return "\n".join(context_parts)
    
    def _fallback_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when the LLM is not available."""
        if not records:
            return "I couldn't find any information matching your query. Please try rephrasing your question."
            
        try:
            # For contractor queries, group by contractor
            if any(term in query.lower() for term in ['contractor', 'company', 'firm']):
                contractors = {}
                for record in records:
                    contractor = record.get('Contractor', 'Unknown Contractor')
                    # Clean up contractor name
                    if not contractor or contractor.strip() == '' or contractor.lower() == 'nan':
                        contractor = 'Unknown Contractor'
                    if contractor not in contractors:
                        contractors[contractor] = []
                    contractors[contractor].append(record)
                
                response = ["I found projects from the following contractors:"]
                for contractor, projects in list(contractors.items())[:5]:  # Limit to top 5
                    response.append(f"- {contractor}: {len(projects)} projects")
                return "\n".join(response)
                
            # For cost queries, show top projects by cost
            if any(term in query.lower() for term in ['cost', 'price', 'budget', 'expensive']):
                sorted_records = sorted(
                    records,
                    key=lambda x: float(x.get('ContractCost', 0)) if x.get('ContractCost') else 0,
                    reverse=True
                )
                response = ["Here are the most expensive projects I found:"]
                for i, record in enumerate(sorted_records[:5], 1):
                    name = record.get('ProjectDescription', 'Unnamed Project')
                    cost = record.get('ContractCost', 0)
                    location = f"{record.get('Municipality', '')}, {record.get('Province', '')}"
                    if isinstance(cost, (int, float)) and cost > 0:
                        cost_str = f"₱{cost:,.2f}"
                    else:
                        cost_str = "Cost not available"
                    response.append(f"{i}. {name} in {location} - {cost_str}")
                return "\n".join(response)
                
            # Default response - use actual dataset column names
            project_details = []
            for r in records[:5]:
                project_name = r.get('ProjectDescription', 'Unnamed Project')
                contractor = r.get('Contractor', 'Unknown Contractor')
                location = f"{r.get('Municipality', '')}, {r.get('Province', '')}"
                cost = r.get('ContractCost', 0)
                
                if isinstance(cost, (int, float)) and cost > 0:
                    cost_str = f" - ₱{cost:,.2f}"
                else:
                    cost_str = ""
                
                project_details.append(f"• {project_name} in {location} by {contractor}{cost_str}")
            
            return f"I found {len(records)} relevant projects. Here are some details:\n\n" + "\n".join(project_details)
                   
        except Exception as e:
            logger.error(f"Error in fallback response: {str(e)}")
            return "I found some information but couldn't process it properly. Please try rephrasing your question."
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_seconds_between_calls:
            time_to_wait = self.min_seconds_between_calls - time_since_last_call
            time.sleep(time_to_wait)
            
        self.last_api_call = time.time()
    
    def generate_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate response using TinyLlama to analyze flood control project data.
        
        Args:
            query: The user's query about flood control projects
            results: List of relevant project records from the CSV dataset
            
        Returns:
            str: TinyLlama's analysis and response
        """
        if not self.is_available():
            return self._fallback_response(query, results)
            
        try:
            # Prepare project data for TinyLlama to analyze
            context = self._prepare_context(results)
            
            # Create a comprehensive prompt for TinyLlama
            prompt = f"""FLOOD CONTROL QUERY: {query}

{context}

Please analyze this flood control project data and provide a comprehensive response to the user's query. Focus on the specific information they're asking about and provide insights, patterns, and actionable information based on the data."""
            
            # Set parameters optimized for TinyLlama
            max_tokens = min(1000, 500 + len(results) * 20)  # Scale with data size
            temperature = 0.7  # Allow some creativity for insights
            
            # Generate response using TinyLlama
            response_text = self._generate_huggingface_response(prompt, "general", max_tokens, temperature)
            
            # Clean up response and add attribution
            response_text = response_text.strip()
            if response_text and len(results) > 0:
                if "based on" not in response_text.lower():
                    response_text += f"\n\n*📊 Analysis based on {len(results)} flood control project(s) from the DPWH dataset.*"
            
            return response_text if response_text else self._fallback_response(query, results)
            
        except Exception as e:
            logger.error(f"Error generating TinyLlama response: {e}")
            return self._fallback_response(query, results)
    
    
    def _generate_huggingface_response(self, prompt: str, query_type: str, max_tokens: int, temperature: float) -> str:
        """Generate response using TinyLlama following official Hugging Face chat template."""
        try:
            logger.info(f"Generating TinyLlama response with {max_tokens} max tokens")
            
            # Create messages following TinyLlama's official chat format
            system_prompt = self._get_system_prompt(query_type)
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Use tokenizer's chat template as recommended by TinyLlama
            formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate response using TinyLlama's recommended parameters
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                return_full_text=False
            )
            
            # Extract the generated text
            generated_text = outputs[0]['generated_text'].strip()
            
            # Clean up any remaining template tokens
            generated_text = re.sub(r'<\|.*?\|>', '', generated_text)
            generated_text = re.sub(r'</s>', '', generated_text)
            generated_text = generated_text.strip()
            
            # If response is too short, try with adjusted parameters
            if len(generated_text) < 50:
                logger.warning("Generated response too short, retrying with adjusted parameters")
                outputs = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=max_tokens + 100,
                    do_sample=True,
                    temperature=min(temperature + 0.1, 0.9),
                    top_k=40,
                    top_p=0.9,
                    return_full_text=False
                )
                generated_text = outputs[0]['generated_text'].strip()
                generated_text = re.sub(r'<\|.*?\|>', '', generated_text)
                generated_text = re.sub(r'</s>', '', generated_text)
                generated_text = generated_text.strip()
            
            return generated_text if generated_text else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating TinyLlama response: {e}")
            # Extract query from prompt for fallback
            query = "flood control query"
            if "FLOOD CONTROL QUERY: " in prompt:
                query = prompt.split("FLOOD CONTROL QUERY: ")[1].split("\n")[0]
            return self._fallback_response(query, [])
    
    def is_available(self) -> bool:
        """Check if TinyLlama model is available."""
        return self.pipeline is not None and self.available
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "provider": "Hugging Face",
            "available": str(self.is_available()),
            "type": "huggingface"
        }
