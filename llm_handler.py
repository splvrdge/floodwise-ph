import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import httpx
import openai
import streamlit as st
from dotenv import load_dotenv
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QueryType(Enum):
    """Types of user queries that can be handled."""
    GENERAL_CONVERSATION = "general_conversation"
    FLOOD_CONTROL_QUERY = "flood_control_query"
    DATA_ANALYSIS = "data_analysis"
    UNKNOWN = "unknown"

class LLMHandler:
    """Handles LLM interactions for generating responses about flood control projects."""
    
    # Common greetings and small talk phrases
    GENERAL_PHRASES = [
        "hi", "hello", "hey", "how are you", "what's up", "good morning", 
        "good afternoon", "good evening", "hi there", "hey there", "greetings"
    ]
    
    # Keywords that indicate a flood control query
    FLOOD_KEYWORDS = [
        "flood", "drainage", "control", "project", "contract", "cost",
        "location", "region", "province", "municipality", "contractor",
        "flood control", "flood mitigation", "drainage system"
    ]
    
    def __init__(self):
        self.client = None
        self.model = "gpt-3.5-turbo"
        self._initialize_client()
    
    def detect_query_type(self, query: str) -> QueryType:
        """
        Determine the type of the user query using a hybrid approach.
        
        Args:
            query: The user's input query
            
        Returns:
            QueryType: The detected type of query
        """
        if not query or not query.strip():
            return QueryType.GENERAL_CONVERSATION
            
        query_lower = query.lower().strip()
        
        # First check for general conversation patterns
        if any(phrase in query_lower.split() for phrase in ["hi", "hello", "hey", "greetings"]):
            return QueryType.GENERAL_CONVERSATION
            
        # Check for how are you and similar greetings
        if any(phrase in query_lower for phrase in ["how are you", "what's up", "how's it going"]):
            return QueryType.GENERAL_CONVERSATION
            
        # Check for thanks/bye patterns
        if any(phrase in query_lower for phrase in ["thank", "thanks", "bye", "goodbye"]):
            return QueryType.GENERAL_CONVERSATION
            
        # Check for flood control related queries
        flood_terms = ["flood", "drainage", "project", "contract", "contractor", 
                      "location", "region", "province", "municipality", "cost",
                      "budget", "construction", "mitigation"]
                      
        if any(term in query_lower for term in flood_terms):
            # Check if it's a data analysis query
            analysis_terms = ["analyze", "analysis", "trend", "statistic", 
                            "compare", "summary", "overview", "insight"]
            if any(term in query_lower for term in analysis_terms):
                return QueryType.DATA_ANALYSIS
            return QueryType.FLOOD_CONTROL_QUERY
            
        # If we can't determine, use LLM to classify
        return self._classify_query_with_llm(query)
    
    def _classify_query_with_llm(self, query: str) -> QueryType:
        """Use the LLM to classify ambiguous queries."""
        if not self.client:
            logger.warning("OpenAI client not initialized")
            return QueryType.UNKNOWN
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies user queries. "
                                                "Respond with just one of: 'general', 'flood_control', or 'data_analysis'"},
                    {"role": "user", "content": f"Classify this query: {query}"}
                ],
                temperature=0.0
            )
            
            classification = response.choices[0].message.content.lower()
            
            if 'general' in classification:
                return QueryType.GENERAL_CONVERSATION
            elif 'data_analysis' in classification or 'analysis' in classification:
                return QueryType.DATA_ANALYSIS
            else:
                return QueryType.FLOOD_CONTROL_QUERY
                
        except Exception as e:
            logger.warning(f"Error classifying query with LLM: {e}")
            return QueryType.UNKNOWN
    
    def handle_general_conversation(self, query: str) -> str:
        """Handle general conversation queries with natural responses."""
        query_lower = query.lower().strip()
        
        # Handle common greetings and small talk
        if any(phrase in query_lower for phrase in ["hi", "hello", "hey", "greetings"]):
            return "Hello! I'm here to help you explore flood control projects in the Philippines. What would you like to know?"
            
        if any(phrase in query_lower for phrase in ["how are you", "how's it going"]):
            return "I'm doing well, thank you for asking! I'm ready to help you find information about flood control projects. What would you like to know?"
            
        if any(phrase in query_lower for phrase in ["thank", "thanks"]):
            return "You're welcome! Is there anything else you'd like to know about flood control projects?"
            
        if any(phrase in query_lower for phrase in ["bye", "goodbye"]):
            return "Goodbye! Feel free to come back if you have more questions about flood control projects."
            
        if any(phrase in query_lower for phrase in ["help", "what can you do"]):
            return ("I can help you find information about flood control projects across the Philippines. "
                   "You can ask me about projects by location, cost, contractor, or completion status. "
                   "For example, you could ask 'Show me flood control projects in Cebu' or 'What are the most expensive projects?'")
        
        # For other general queries, use the LLM with a focused prompt
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant for the Flood Control Projects Database. "
                                 "Respond naturally and conversationally. If the query is related to flood control "
                                 "projects in the Philippines, say you can help with that. Otherwise, keep it brief. "
                                 "Be friendly and professional."
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in general conversation: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Could you try asking about flood control projects in the Philippines?"
    
    def generate_response(self, query: str, records: List[Dict[str, Any]] = None, context: Dict[str, Any] = None) -> str:
        """
        Generate a response based on the user's query and available records using a hybrid approach.
        
        Args:
            query: The user's input query
            records: Optional list of records from the database (for flood control queries)
            context: Additional context for the response
            
        Returns:
            str: The generated response in natural language
        """
        if not query or not query.strip():
            return "I didn't catch that. Could you please rephrase your question?"
        
        # Initialize records if not provided
        if records is None:
            records = []
            
        # Detect the type of query
        query_type = self.detect_query_type(query)
        logger.info(f"Detected query type: {query_type}")
        
        # Handle general conversation
        if query_type == QueryType.GENERAL_CONVERSATION:
            return self.handle_general_conversation(query)
            
        # Handle unknown query types
        if query_type == QueryType.UNKNOWN:
            return ("I'm not sure if this is related to flood control projects. "
                   "I can help you find information about flood control projects in the Philippines, "
                   "including details about costs, locations, contractors, and project status. "
                   "Could you rephrase your question?")
        
        # For flood control queries, check if we have records
        if not records and query_type != QueryType.GENERAL_CONVERSATION:
            return self._generate_no_results_response(query)
            
        # Prepare context if not provided
        if context is None:
            context = {}
        
        # Use LLM to generate a natural response based on the data
        try:
            # Format the records into a more natural, readable string
            records_str = "\n".join([
                f"- {record.get('ProjectDescription', 'A flood control project')} "
                f"in {record.get('Municipality', 'an unspecified location')}, "
                f"{record.get('Province', 'an unspecified province')} "
                f"(Cost: ₱{float(record.get('ContractCost', 0)):,.2f}, "
                f"Contractor: {record.get('Contractor', 'Not specified')}, "
                f"Status: {record.get('Status', 'Unknown')})"
                for record in records[:5]  # Limit to first 5 records
            ])
            
            # Generate a natural language response using the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that provides information about flood control projects in the Philippines. "
                            "Use the provided project data to answer the user's question in a clear, natural way. "
                            "Be concise but informative. If the data doesn't contain the exact information requested, "
                            "say so and provide what information is available. Format numbers clearly (e.g., ₱1,234,567.89)."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {query}\n\n"
                            f"Here are some relevant projects from our database:\n{records_str}\n\n"
                            "Please provide a helpful, natural response based on this information. "
                            "If the question asks for specific details not in the data, explain what information is available. "
                            "Use markdown for formatting if helpful, but keep it simple."
                        )
                    }
                ],
                temperature=0.3,
                max_tokens=500,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fall back to structured response if LLM fails
            try:
                response_type = self._determine_response_type(query, records)
                if response_type == "cost":
                    return self._format_cost_response(query, records)
                elif response_type == "contractor":
                    return self._format_contractor_response(query, records)
                elif response_type == "location":
                    return self._format_standard_location_response(query, records)
                else:
                    return self._format_general_response(query, records)
            except Exception as inner_e:
                logger.error(f"Fallback response generation failed: {inner_e}")
                return "I'm having trouble processing your request right now. Please try again later or rephrase your question."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            try:
                # Try a simple fallback using the LLM directly
                prompt = self._prepare_prompt(query, records, context, query_type)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                    top_p=0.9,
                    frequency_penalty=0.5,
                    presence_penalty=0.5
                )
                return response.choices[0].message.content.strip()
            except Exception as inner_e:
                logger.error(f"Fallback response generation also failed: {inner_e}")
                return self._fallback_response(query, records)
    
    def _initialize_client(self):
        """Initialize OpenAI client with multiple fallback strategies."""
        api_key = None
        
        # Try to get API key from environment variables first
        api_key = os.getenv("OPENAI_API_KEY")
        
        # If not in environment, try Streamlit secrets
        if not api_key and hasattr(st, 'secrets') and st.secrets:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY")
                # Show available keys for debugging
                available_keys = list(st.secrets.keys()) if st.secrets else []
                logger.debug(f"Available secret keys: {available_keys}")
            except Exception as e:
                logger.warning(f"Error accessing Streamlit secrets: {e}")
        
        # If still no API key, log a warning and return
        if not api_key:
            logger.warning("No OpenAI API key found. Some features may not work properly.")
            self.client = None
            return
        
        # Try multiple initialization strategies to handle different environments
        strategies = [
            ("Default Client", self._init_with_default_client),
            ("Custom HTTP Client", self._init_with_custom_http_client),
            ("Legacy Client", self._init_with_legacy_approach)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Trying to initialize with {strategy_name}...")
                strategy_func(api_key)
                logger.info(f"Successfully initialized with {strategy_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize with {strategy_name}: {e}")
        
        # If all initialization methods fail
        logger.error("All OpenAI client initialization methods failed")
        self.client = None
        
        # Log debug info if debug mode is enabled
        if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
            st.warning("Debug: All OpenAI initialization strategies failed. Running in basic mode.")
    
    def _init_with_custom_http_client(self, api_key: str):
        """Initialize with custom HTTP client to avoid proxies issue."""
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
    
    def _get_system_prompt(self, query_type: str) -> str:
        """Get the appropriate system prompt based on query type."""
        base_prompt = """You are a knowledgeable assistant providing information about flood control projects in the Philippines. 
        Your responses should be based SOLELY on the provided dataset. If information isn't available in the data, 
        clearly state that rather than making assumptions.
        
        Response Guidelines:
        - Use natural, conversational language
        - Be concise but informative
        - Use bullet points or numbered lists only when necessary
        - Vary your sentence structure
        - Use contractions (e.g., "don't" instead of "do not")
        - If exact numbers aren't available, be clear about what information you do have
        - Never make up or hallucinate information
        - If the query is unclear, ask for clarification
        - Format monetary values clearly (e.g., ₱1,234,567.89)
        - For location-based queries, be specific about the administrative divisions (region, province, city/municipality)
        - For contractor information, focus on the specific projects from the dataset
        - For cost-related queries, be clear about whether the amounts are in Philippine Pesos (₱)
        - For project types, use the exact terminology from the dataset
        - For time-based queries, be clear about the time periods being referenced
        - If multiple records are relevant, summarize the key points rather than listing each one
        - When appropriate, provide context about why certain information might be significant
        - If the query is too broad, suggest ways to narrow it down
        - Always maintain a helpful and professional tone
        - If you need to reference multiple records, group them logically
        - When discussing costs, be clear about whether they're totals or averages"""
        
        type_specific = {
            'location': 'Focus on the geographic aspects of the projects, including regions, provinces, and municipalities. Highlight any patterns in project distribution.',
            'time': 'Concentrate on project timelines, including start and completion dates. Note any trends or patterns in project duration.',
            'cost': 'Provide clear information about project costs, including contract amounts and any available budget breakdowns. Highlight any significant investments.',
            'contractor': 'Discuss the contractors involved in the projects, focusing on their specific roles and the scale of their involvement. Only include information available in the dataset.',
            'type': 'Describe the different types of flood control projects, using the exact terminology from the dataset. Note any variations in project scope or implementation.',
            'general': 'Provide a balanced overview of the flood control projects, focusing on the key details that would be most relevant to the query.'
        }
        
        return f"{base_prompt}\n\n{type_specific.get(query_type, type_specific['general'])}"

    def _prepare_prompt(self, query: str, records: List[Dict[str, Any]], context: Dict[str, Any] = None, query_type: str = 'general') -> str:
        """
        Prepare the prompt for the LLM based on the query and records.
        
        Args:
            query: The user's query
            records: List of relevant records from the database
            context: Additional context for the response
            query_type: Type of the query (general, cost, contractor, etc.)
            
        Returns:
            str: The generated prompt
        """
        try:
            # Format the records into a more natural, readable string
            records_str = "\n".join([
                f"- {record.get('ProjectDescription', 'A flood control project')} "
                f"in {record.get('Municipality', 'an unspecified location')}, "
                f"{record.get('Province', 'an unspecified province')} "
                f"(Cost: ₱{float(record.get('ContractCost', 0)):,.2f}, "
                f"Contractor: {record.get('Contractor', 'Not specified')}, "
                f"Status: {record.get('Status', 'Unknown')})"
                for record in records[:10]  # Limit to first 10 records to avoid overwhelming the context
            ])
            
            # Include context if available
            context_str = ""
            if context:
                context_str = "\n\nAdditional context: "
                for key, value in context.items():
                    context_str += f"\n- {key}: {value}"
            
            # Get the system prompt based on query type
            system_prompt = self._get_system_prompt(query_type)
            
            # Create the full prompt
            prompt = f"""{system_prompt}

User Query: {query}

Available project information:
{records_str}{context_str}

Please provide a helpful, accurate response based on the information above. If the data doesn't contain the specific details being asked about, please let the user know what information is available."""
            
            return prompt
                
        except Exception as e:
            # If there's an error with the LLM, fall back to a simpler response
            print(f"Error generating response: {str(e)}")
            return self._fallback_response(query, records)
            
    def _fallback_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when the LLM is not available."""
        if not records:
            return "I couldn't find any information matching your query. Please try rephrasing your question."
            
        try:
            # For contractor queries, group by contractor
            if any(term in query.lower() for term in ['contractor', 'company', 'firm']):
                contractors = {}
                for record in records:
                    contractor = record.get('contractor', 'Unknown')
                    if contractor not in contractors:
                        contractors[contractor] = []
                    contractors[contractor].append(record)
                
                response = ["I found projects from the following contractors:"]
                for contractor, projects in list(contractors.items())[:5]:  # Limit to top 5
                    response.append(f"- {contractor}: {len(projects)} projects")
                return "\n".join(response)
                
            # For cost queries, show top projects by cost
            if any(term in query.lower() for term in ['cost', 'price', 'budget']):
                sorted_records = sorted(
                    records,
                    key=lambda x: float(x.get('contract_cost', 0)) if x.get('contract_cost') else 0,
                    reverse=True
                )
                response = ["Here are the most expensive projects I found:"]
                for i, record in enumerate(sorted_records[:5], 1):
                    name = record.get('project_name', 'Unnamed Project')
                    cost = record.get('contract_cost', 'N/A')
                    response.append(f"{i}. {name} - {cost}")
                return "\n".join(response)
                
            # Default response
            project_details = "\n\n".join(
                f"• {r.get('project_name', 'Unnamed Project')} - {r.get('contractor', 'Unknown')}" 
                for r in records[:5]
            )
            return f"I found {len(records)} relevant projects. Here are some details:\n\n{project_details}"
                   
        except Exception as e:
            print(f"Error in fallback response: {str(e)}")
            return "I found some information but couldn't process it properly. Please try rephrasing your question."
    
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
            record_info = []
            for key, value in record.items():
                if value is not None and value != '':
                    if key.lower() == 'contractcost' and isinstance(value, (int, float, str)):
                        try:
                            cost = float(value)
                            value = f"₱{cost:,.2f}"
                        except (ValueError, TypeError):
                            pass
                    record_info.append(f"  - {key}: {value}")
            context_parts.extend(record_info)
            
        return "\n".join(context_parts)
    
    def _format_cost_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a natural language response about project costs."""
        if not records:
            return "I couldn't find any cost information in the dataset."
            
        # Check if this is a cost summary query
        if records and records[0].get('query_type') == 'cost_summary':
            return self._format_cost_summary_response(query, records)
        
        query_lower = query.lower()
        
        # Check for "most expensive" pattern
        is_most_expensive_query = any(term in query_lower for term in ["most expensive", "highest cost", "largest budget"])
        
        # Extract location from query if present
        location_terms = ['in ', 'at ', 'for ', 'near ']
        location = None
        
        for term in location_terms:
            if term in query_lower:
                # Get the text after the location term
                loc_start = query_lower.find(term) + len(term)
                location = query[loc_start:].split('.')[0].split('?')[0].strip()
                break
        
        # Filter out records without cost and sort by cost (highest first)
        valid_records = [r for r in records if r.get('ContractCost') and str(r.get('ContractCost', '')).strip().lower() not in ['', 'n/a', 'none']]
        
        if not valid_records:
            if location:
                return f"I couldn't find any projects with cost information in {location}."
            return "I couldn't find any projects with cost information in the dataset."
        
        valid_records.sort(key=lambda x: float(str(x.get('ContractCost', '0')).replace(',', '')), reverse=True)
        
        # If location was specified, filter records for that location
        if location:
            location_records = []
            for r in valid_records:
                if (location.lower() in str(r.get('Municipality', '')).lower() or 
                    location.lower() in str(r.get('Province', '')).lower() or
                    location.lower() in str(r.get('Region', '')).lower() or
                    any(location.lower() in loc.lower() for loc in [r.get('Municipality', ''), r.get('Province', ''), r.get('Region', '') if r.get('Region') else ''])):
                    location_records.append(r)
            
            if location_records:
                valid_records = location_records
                intro = f"Here are the flood control projects in {location} sorted by budget (highest first):"
            else:
                return f"I couldn't find any flood control projects in {location}. Would you like to try a different location?"
        else:
            intro = "Here are some flood control projects sorted by budget (highest first):"
        
        # If it's a "most expensive" query, just return the top result
        if is_most_expensive_query and valid_records:
            record = valid_records[0]  # Already sorted by cost, highest first
            cost = float(str(record.get('ContractCost', '0')).replace(',', ''))
            
            location_info = []
            if record.get('Municipality'):
                location_info.append(record['Municipality'])
            if record.get('Province') and record['Province'].lower() not in [loc.lower() for loc in location_info]:
                location_info.append(record['Province'])
            
            response = f"## 🏆 Most Expensive Project"
            if location:
                response += f" in {location}"
            response += "\n\n"
            
            response += (
                f"### {record.get('ProjectDescription', 'Flood Control Project')}\n"
                f"📍 **Location:** {', '.join(location_info) or 'Not specified'}\n"
                f"💰 **Budget:** ₱{cost:,.2f}\n"
            )
            
            # Add contractor and status if available
            if record.get('Contractor') and str(record['Contractor']).strip().lower() not in ['', 'n/a']:
                response += f"🏗️ **Contractor:** {record['Contractor']}\n"
            if record.get('Status') and str(record['Status']).strip().lower() not in ['', 'n/a']:
                status = str(record['Status']).strip()
                status_emoji = "✅" if 'complete' in status.lower() else "🔄"
                response += f"{status_emoji} **Status:** {status}\n"
            
            # Add completion year if available
            if (record.get('CompletionYear') and 
                str(record['CompletionYear']).strip().lower() not in ['', 'n/a', 'none']):
                response += f"📅 **Completed:** {record['CompletionYear']}\n"
            
            # Add a note if there are more projects
            if len(valid_records) > 1:
                response += f"\n*Note: There are {len(valid_records) - 1} other projects in this location.*"
            
            return response
        
        # For non-"most expensive" queries, return multiple results
        # Calculate statistics
        costs = [float(str(r.get('ContractCost', '0')).replace(',', '')) for r in valid_records]
        total_cost = sum(costs)
        avg_cost = total_cost / len(costs) if costs else 0
        
        # Build the response
        response = f"## 📊 {intro}\n\n"
        
        # Add summary statistics if we have multiple projects
        if len(valid_records) > 1:
            response += (f"- **Total projects found:** {len(valid_records)}\n"
                        f"- **Total investment:** ₱{total_cost:,.2f}\n"
                        f"- **Average project cost:** ₱{avg_cost:,.2f}\n\n")
        
        # Add project details
        for i, record in enumerate(valid_records[:5], 1):  # Limit to top 5 for brevity
            cost = float(str(record.get('ContractCost', '0')).replace(',', ''))
            location_info = []
            if record.get('Municipality'):
                location_info.append(record['Municipality'])
            if record.get('Province') and record['Province'].lower() not in [loc.lower() for loc in location_info]:
                location_info.append(record['Province'])
            
            response += (
                f"### {i}. {record.get('ProjectDescription', 'Flood Control Project')}\n"
                f"📍 **Location:** {', '.join(location_info) or 'Not specified'}\n"
                f"💰 **Budget:** ₱{cost:,.2f}\n"
            )
            
            # Add contractor and status if available
            if record.get('Contractor') and str(record['Contractor']).strip().lower() not in ['', 'n/a']:
                response += f"🏗️ **Contractor:** {record['Contractor']}\n"
            if record.get('Status') and str(record['Status']).strip().lower() not in ['', 'n/a']:
                status = str(record['Status']).strip()
                status_emoji = "✅" if 'complete' in status.lower() else "🔄"
                response += f"{status_emoji} **Status:** {status}\n"
            
            # Add completion year if available
            if (record.get('CompletionYear') and 
                str(record['CompletionYear']).strip().lower() not in ['', 'n/a', 'none']):
                response += f"📅 **Completed:** {record['CompletionYear']}\n"
            
            response += "\n"  # Add spacing between projects
        
        # Add a note about additional results
        if len(valid_records) > 5:
            response += f"*Note: Showing top 5 projects out of {len(valid_records)}. "
            response += "You can ask for more specific details about any project.*"
        
        return response
    
    def _format_cost_summary_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for cost summary queries (average, total, etc.)."""
        if not records:
            return "No cost data available."
        
        # Get summary statistics from the first record
        first_record = records[0]
        total_investment = first_record.get('total_investment', 0)
        average_cost = first_record.get('average_cost', 0)
        project_count = first_record.get('project_count', 0)
        
        response_parts = []
        
        if 'average' in query.lower():
            response_parts.append("# 📊 Average Spending Analysis")
        elif 'total' in query.lower():
            response_parts.append("# 💰 Total Investment Analysis")
        else:
            response_parts.append("# 📈 Cost Summary Analysis")
        
        response_parts.append("")
        response_parts.append("## 📊 Summary Statistics")
        response_parts.append(f"**💰 Total Investment:** ₱{total_investment:,.2f}")
        response_parts.append(f"**📈 Average Cost per Project:** ₱{average_cost:,.2f}")
        response_parts.append(f"**📊 Number of Projects:** {project_count}")
        response_parts.append("")
        
        # Show sample projects
        response_parts.append("## 🏗️ Sample Projects")
        for i, record in enumerate(records[:3], 1):
            cost = float(record.get('ContractCost', 0))
            cost_vs_avg = ((cost - average_cost) / average_cost * 100) if average_cost > 0 else 0
            
            response_parts.append(f"### {i}. {record.get('ProjectDescription', 'Unknown Project')}")
            response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
            response_parts.append(f"**💰 Contract Cost:** ₱{cost:,.2f} ({cost_vs_avg:+.1f}% vs average)")
            response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
            response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
            response_parts.append("")
        
        if len(records) > 3:
            response_parts.append(f"*Showing 3 sample projects out of {project_count} total projects.*")
        
        return "\n".join(response_parts)
    
    def _format_contractor_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a natural language response about contractors and their projects."""
        if not records:
            return "I couldn't find any information about contractors in the dataset."
        
        # Extract location from query if present
        query_lower = query.lower()
        location_terms = ['in ', 'at ', 'for ', 'near ']
        location = None
        
        for term in location_terms:
            if term in query_lower:
                # Get the text after the location term
                loc_start = query_lower.find(term) + len(term)
                location = query[loc_start:].split('.')[0].split('?')[0].strip()
                break
        
        # Filter records by location if specified
        if location:
            location_records = []
            for r in records:
                if (location.lower() in str(r.get('Municipality', '')).lower() or 
                    location.lower() in str(r.get('Province', '')).lower() or
                    location.lower() in str(r.get('Region', '')).lower()):
                    location_records.append(r)
            
            if not location_records:
                return f"I couldn't find any contractors with projects in {location}. Would you like to try a different location?"
            
            records = location_records
        
        # Create a dictionary to group projects by contractor
        contractors = {}
        for record in records:
            contractor = record.get('Contractor', 'Unknown Contractor')
            if contractor.lower() == 'n/a' or not contractor.strip():
                contractor = 'Unknown Contractor'
                
            if contractor not in contractors:
                contractors[contractor] = {
                    'projects': [],
                    'total_value': 0,
                    'locations': set(),
                    'recent_year': 0
                }
            
            cost = float(record.get('ContractCost', 0))
            year = int(record['CompletionYear']) if record.get('CompletionYear') and str(record['CompletionYear']).isdigit() else 0
            location = f"{record.get('Municipality', '')}, {record.get('Province', '')}".strip(', ')
            
            contractors[contractor]['projects'].append({
                'description': record.get('ProjectDescription', 'Flood Control Project'),
                'cost': cost,
                'location': location,
                'year': year
            })
            
            contractors[contractor]['total_value'] += cost
            if location:
                contractors[contractor]['locations'].add(location)
            if year > contractors[contractor]['recent_year']:
                contractors[contractor]['recent_year'] = year
        
        # Sort contractors by number of projects (descending) and then by total value (descending)
        sorted_contractors = sorted(
            contractors.items(),
            key=lambda x: (len(x[1]['projects']), x[1]['total_value']),
            reverse=True
        )
        
        # Build the response
        response = "## 🏗️ "
        if location:
            response += f"Top Contractors in {location}\n\n"
        else:
            response += "Top Contractors\n\n"
        
        response += "Here are the most active contractors based on the number of projects and total contract value.\n\n"
        
        # Add contractor details
        for i, (contractor, data) in enumerate(sorted_contractors[:10], 1):  # Limit to top 10
            project_count = len(data['projects'])
            total_value = data['total_value']
            locations = ", ".join(sorted(data['locations'])[:3])  # Show up to 3 locations
            if len(data['locations']) > 3:
                locations += f" and {len(data['locations']) - 3} more locations"
            
            # Get the most expensive project
            most_expensive = max(data['projects'], key=lambda x: x['cost'])
            
            response += (
                f"### {i}. {contractor}\n"
                f"📊 **Projects:** {project_count} • 💰 **Total Value:** ₱{total_value:,.2f}\n"
            )
            
            if locations:
                response += f"📍 **Locations:** {locations}\n"
                
            response += (
                f"🏆 **Most Expensive Project:** {most_expensive['description']} "
                f"(₱{most_expensive['cost']:,.2f} in {most_expensive['year'] or 'N/A'})\n\n"
            )
        
        # Add a note about additional results
        if len(sorted_contractors) > 10:
            response += f"\n*Note: Showing top 10 out of {len(sorted_contractors)} contractors. "
            response += "You can ask for more specific information about any contractor or location.*"
        
        return response
    
    def _format_completion_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for completion-related queries."""
        response_parts = ["# 📅 Project Completion Analysis", ""]
        
        # Group by year if multiple years present
        years = set(str(r.get('CompletionYear', 'N/A')) for r in records)
        if len(years) > 1:
            response_parts.append(f"**📊 Years Covered:** {', '.join(sorted(years))}")
            response_parts.append("")
        
        for i, record in enumerate(records[:5], 1):
            completion_year = record.get('CompletionYear', 'N/A')
            completion_date = record.get('CompletionDateActual', 'N/A')
            start_date = record.get('StartDate', 'N/A')
            
            response_parts.append(f"## {i}. {record.get('ProjectDescription', 'Unknown Project')}")
            response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
            response_parts.append(f"**📅 Completion Year:** {completion_year}")
            
            if completion_date != 'N/A':
                response_parts.append(f"**🗓️ Completion Date:** {completion_date}")
            if start_date != 'N/A':
                response_parts.append(f"**🚀 Start Date:** {start_date}")
            
            response_parts.append(f"**💰 Contract Cost:** ₱{float(record.get('ContractCost', 0)):,.2f}")
            response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _format_project_type_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """
        Format response for project type queries with comprehensive categorization.
        
        Args:
            query: The original user query
            records: List of project records matching the query
            
        Returns:
            Formatted response string with categorized project types
        """
        if not records:
            return "I couldn't find any information about project types in the database."
            
        response = ["# 🌊 Types of Flood Control Projects\n"]
        
        # Categorize projects into standard flood control types
        project_categories = {
            'Structural Measures': {
                'Flood Walls and Levees': [],
                'Retention/Detention Basins': [],
                'River Channel Improvements': [],
                'Drainage Systems': [],
                'Coastal Protection': []
            },
            'Non-Structural Measures': {
                'Flood Forecasting Systems': [],
                'Watershed Management': [],
                'Floodplain Mapping': [],
                'Community Awareness Programs': []
            },
            'Other Projects': []
        }
        
        # Categorize each project based on keywords in the project name or description
        for record in records:
            project_desc = str(record.get('ProjectDescription', '')).lower()
            project_type = str(record.get('TypeofWork', '')).lower()
            
            # Check for structural measures
            if any(term in project_desc + ' ' + project_type for term in ['wall', 'levee', 'dike', 'embankment']):
                project_categories['Structural Measures']['Flood Walls and Levees'].append(record)
            elif any(term in project_desc + ' ' + project_type for term in ['basin', 'retention', 'detention']):
                project_categories['Structural Measures']['Retention/Detention Basins'].append(record)
            elif any(term in project_desc + ' ' + project_type for term in ['river', 'channel', 'dredg', 'bank protection']):
                project_categories['Structural Measures']['River Channel Improvements'].append(record)
            elif any(term in project_desc + ' ' + project_type for term in ['drain', 'culvert', 'sewer']):
                project_categories['Structural Measures']['Drainage Systems'].append(record)
            elif any(term in project_desc + ' ' + project_type for term in ['coast', 'shore', 'beach', 'mangrove']):
                project_categories['Structural Measures']['Coastal Protection'].append(record)
                
            # Check for non-structural measures
            elif any(term in project_desc + ' ' + project_type for term in ['forecast', 'monitor', 'early warning']):
                project_categories['Non-Structural Measures']['Flood Forecasting Systems'].append(record)
            elif any(term in project_desc + ' ' + project_type for term in ['watershed', 'reforestation', 'catchment']):
                project_categories['Non-Structural Measures']['Watershed Management'].append(record)
            elif any(term in project_desc + ' ' + project_type for term in ['mapping', 'survey', 'study']):
                project_categories['Non-Structural Measures']['Floodplain Mapping'].append(record)
            elif any(term in project_desc + ' ' + project_type for term in ['training', 'awareness', 'education', 'workshop']):
                project_categories['Non-Structural Measures']['Community Awareness Programs'].append(record)
                
            # If no category matches, add to other projects
            else:
                project_categories['Other Projects'].append(record)
        
        # Build the response with categorized projects
        for category, subcategories in project_categories.items():
            has_items = any(subcategories.values()) if isinstance(subcategories, dict) else bool(subcategories)
            
            if has_items:
                response.append(f"## {category}\n")
                
                if isinstance(subcategories, dict):
                    for subcategory, projects in subcategories.items():
                        if projects:
                            response.append(f"### {subcategory}")
                            for i, project in enumerate(projects[:3], 1):  # Show up to 3 examples per subcategory
                                response.append(f"{i}. **{project.get('ProjectDescription', 'Unnamed Project')}**")
                                if 'Municipality' in project or 'Province' in project:
                                    location = f"{project.get('Municipality', '')}, {project.get('Province', '')}".strip(', ')
                                    if location:
                                        response.append(f"   📍 {location}")
                                if 'ContractCost' in project:
                                    try:
                                        cost = float(project.get('ContractCost', 0))
                                        response.append(f"   💰 Contract Cost: ₱{cost:,.2f}")
                                    except (ValueError, TypeError):
                                        pass
                                if 'Contractor' in project:
                                    response.append(f"   🏗️ Contractor: {project['Contractor']}")
                                response.append("")
                            
                            # Add a note if there are more projects in this category
                            if len(projects) > 3:
                                response.append(f"   *... and {len(projects) - 3} more {subcategory.lower()} projects*\n")
                            else:
                                response.append("")
                else:
                    # Handle uncategorized projects
                    for i, project in enumerate(subcategories[:5], 1):  # Limit to 5 uncategorized projects
                        response.append(f"{i}. **{project.get('ProjectDescription', 'Unnamed Project')}**")
                        if 'TypeofWork' in project:
                            response.append(f"   🔧 Type: {project['TypeofWork']}")
                        if 'Municipality' in project or 'Province' in project:
                            location = f"{project.get('Municipality', '')}, {project.get('Province', '')}".strip(', ')
                            if location:
                                response.append(f"   📍 {location}")
                        if 'ContractCost' in project:
                            try:
                                cost = float(project.get('ContractCost', 0))
                                response.append(f"   💰 Contract Cost: ₱{cost:,.2f}")
                            except (ValueError, TypeError):
                                pass
                        response.append("")
                    
                    if len(subcategories) > 5:
                        response.append(f"*... and {len(subcategories) - 5} more uncategorized projects*\n")
                
                response.append("---\n")
        
        # Add a summary of project counts
        total_projects = sum(len(projects) for subcategories in project_categories.values() 
                           for projects in (subcategories.values() if isinstance(subcategories, dict) else [subcategories]))
        
        response.append(f"### 📊 Summary of Project Types")
        for category, subcategories in project_categories.items():
            if isinstance(subcategories, dict):
                count = sum(len(projects) for projects in subcategories.values())
            else:
                count = len(subcategories)
            if count > 0:
                response.append(f"- **{category}**: {count} projects")
        
        response.append("\n*Note: Project categorization is based on keywords in project names and descriptions. Some projects may appear in multiple categories.*")
        
        return "\n".join(response)
    
    def _format_location_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for location-based queries."""
        # Check if this is a count/investment analysis query
        if records and 'project_count' in records[0]:
            return self._format_location_count_response(query, records)
        elif records and 'total_investment' in records[0]:
            return self._format_location_investment_response(query, records)
        else:
            return self._format_standard_location_response(query, records)
    
    def _format_location_count_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for location count queries."""
        response_parts = ["# 📊 Project Count Analysis by Location", ""]
        
        total_projects = sum(r.get('project_count', 0) for r in records)
        total_investment = sum(r.get('total_investment', 0) for r in records)
        
        response_parts.append(f"**📈 Summary:** {total_projects} projects across {len(records)} locations")
        response_parts.append(f"**💰 Combined Investment:** ₱{total_investment:,.2f}")
        response_parts.append("")
        
        for i, record in enumerate(records[:5], 1):
            response_parts.append(f"## {i}. {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
            response_parts.append(f"**📊 Project Count:** {record.get('project_count', 0)}")
            response_parts.append(f"**💰 Total Investment:** ₱{record.get('total_investment', 0):,.2f}")
            response_parts.append(f"**📈 Average Cost:** ₱{record.get('average_cost', 0):,.2f}")
            
            first_year = record.get('first_project_year')
            last_year = record.get('last_project_year')
            if first_year and last_year:
                if first_year == last_year:
                    response_parts.append(f"**📅 Project Year:** {first_year}")
                else:
                    response_parts.append(f"**📅 Project Period:** {first_year}-{last_year}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _format_location_investment_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for location investment queries."""
        response_parts = ["# 💰 Investment Analysis by Location", ""]
        
        grand_total = sum(r.get('total_investment', 0) for r in records)
        total_projects = sum(r.get('project_count', 0) for r in records)
        
        response_parts.append(f"**🏆 Top Investment Destinations**")
        response_parts.append(f"**💰 Grand Total:** ₱{grand_total:,.2f} across {total_projects} projects")
        response_parts.append("")
        
        for i, record in enumerate(records[:5], 1):
            municipality = record.get('Municipality', 'N/A')
            province = record.get('Province', 'N/A')
            region = record.get('Region', 'N/A')
            
            response_parts.append(f"## {i}. {municipality}, {province}")
            response_parts.append(f"**🌏 Region:** {region}")
            response_parts.append(f"**💰 Total Investment:** ₱{record.get('total_investment', 0):,.2f}")
            response_parts.append(f"**📊 Number of Projects:** {record.get('project_count', 0)}")
            response_parts.append(f"**📈 Average Project Cost:** ₱{record.get('average_project_cost', 0):,.2f}")
            response_parts.append(f"**🏆 Largest Project:** ₱{record.get('largest_project_cost', 0):,.2f}")
            response_parts.append(f"**📅 Investment Period:** {record.get('investment_period', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _format_standard_location_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a natural language response about projects in specific locations."""
        if not records:
            return "I couldn't find any projects in the specified location(s)."
            
        # Get unique locations
        locations = {}
        for r in records:
            loc_key = (r.get('Municipality'), r.get('Province'))
            if loc_key not in locations:
                locations[loc_key] = []
            locations[loc_key].append(r)
            
        response = "Here's what I found about flood control projects "
        
        if len(locations) == 1:
            loc_name = f"{list(locations.keys())[0][0]}, {list(locations.keys())[0][1]}"
            response += f"in {loc_name}:\n\n"
        else:
            response += f"across {len(locations)} different locations. Here are the details:\n\n"
        
        # Calculate statistics
        total_cost = sum(float(r.get('ContractCost', 0)) for r in records if r.get('ContractCost'))
        project_count = len(records)
        avg_cost = total_cost / project_count if project_count > 0 else 0
        
        # Add summary information
        if total_cost > 0:
            response += f"There {'is' if project_count == 1 else 'are'} {project_count} project{'s' if project_count != 1 else ''} "
            response += f"with a total investment of approximately ₱{total_cost:,.2f}. "
            response += f"The average project cost is around ₱{avg_cost:,.2f}.\n\n"
        
        # Group projects by location
        for (municipality, province), loc_records in list(locations.items())[:5]:  # Limit to 5 locations
            loc_name = f"{municipality if municipality else 'Unknown location'}, {province if province else 'unknown province'}"
            loc_cost = sum(float(r.get('ContractCost', 0)) for r in loc_records if r.get('ContractCost'))
            loc_count = len(loc_records)
            
            response += f"In {loc_name}, there {'is' if loc_count == 1 else 'are'} {loc_count} project{'s' if loc_count != 1 else ''} "
            response += f"with a total investment of ₱{loc_cost:,.2f}. "
            
            # Add some project examples
            if loc_count <= 3:
                projects = ", ".join(f"{r.get('ProjectDescription', 'a flood control project')} (₱{float(r.get('ContractCost', 0)):,.2f})" 
                                    for r in loc_records[:3])
                response += f"These include: {projects}."
            else:
                sample = ", ".join(f"{r.get('ProjectDescription', 'a project')} (₱{float(r.get('ContractCost', 0)):,.2f})" 
                                  for r in loc_records[:2])
                response += f"These include {sample}, and {loc_count - 2} more projects."
            
            response += "\n\n"
        
        # Add a note if there are more locations
        if len(locations) > 5:
            response += f"There are {len(locations) - 5} additional locations with flood control projects. "
            response += "Would you like me to provide more details about any specific location?"
        
        return response
    
    def _format_comparison_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a natural language comparison of different entities based on project data."""
        if not records:
            return "I couldn't find enough data to make a meaningful comparison."
        
        # Check if this is a category-based comparison
        if records and 'comparison_category' in records[0]:
            category = records[0].get('comparison_category', 'category')
            response = f"Here's a comparison of projects by {category}:\n\n"
            
            for i, record in enumerate(records[:5], 1):
                category_value = record.get('Region', record.get('Province', record.get('Municipality', f'Item {i}')))
                cost = float(record.get('ContractCost', 0))
                
                response += f"{i}. {category_value}: "
                response += f"{record.get('ProjectDescription', 'A flood control project')} "
                response += f"(₱{cost:,.2f})\n"
                
                if i < min(5, len(records)):  # Add spacing between items
                    response += "\n"
            
            if len(records) > 5:
                response += f"\nThere are {len(records) - 5} more items in this category. "
                response += "Would you like me to show more details about any specific one?"
                
            return response
            
        # Default comparison by grouping records
        comparison_groups = {}
        for record in records:
            group_key = record.get('comparison_term', 'Unknown')
            if group_key not in comparison_groups:
                comparison_groups[group_key] = []
            comparison_groups[group_key].append(record)
        
        # Start building the response
        response = "Here's a comparison based on your query:\n\n"
        
        # Add summary statistics for each group
        for group, items in comparison_groups.items():
            total_cost = sum(float(r.get('ContractCost', 0)) for r in items if r.get('ContractCost'))
            avg_cost = total_cost / len(items) if items else 0
            
            response += f"## {group.title() if isinstance(group, str) else 'Comparison Group'}\n"
            response += f"- Number of projects: {len(items)}"
            response += f"\n- Total investment: ₱{total_cost:,.2f}"
            response += f"\n- Average project cost: ₱{avg_cost:,.2f}"
            
            # Add a sample project
            if items:
                sample = items[0]
                response += f"\n- Sample project: {sample.get('ProjectDescription', 'A flood control project')} "
                response += f"in {sample.get('Municipality', 'an unspecified location')}, "
                response += f"{sample.get('Province', 'an unspecified province')} "
                response += f"(₱{float(sample.get('ContractCost', 0)):,.2f})"
            
            response += "\n\n"
        
        # Add a closing note if there are many groups
        if len(comparison_groups) > 3:
            response += "This is a high-level comparison. "
            response += "Would you like me to provide more details about any specific group?"
        
        return response
    
    def _format_general_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a natural language response for general queries about flood control projects."""
        if not records:
            return "I couldn't find any projects matching your query. Could you try being more specific or check your search terms?"
        
        # Start building the response
        response = f"I found {len(records)} flood control project{'s' if len(records) != 1 else ''} "
        response += f"related to your query. Here's a summary:\n\n"
        
        # Group by location if there are multiple locations
        locations = {}
        for record in records:
            loc_key = (record.get('Municipality'), record.get('Province'))
            if loc_key not in locations:
                locations[loc_key] = []
            locations[loc_key].append(record)
        
        # If all projects are in the same location, mention it once
        if len(locations) == 1:
            loc_name = f"{list(locations.keys())[0][0] if list(locations.keys())[0][0] else 'various locations'}, {list(locations.keys())[0][1] if list(locations.keys())[0][1] else 'various provinces'}"
            response += f"All projects are located in {loc_name}. "
        else:
            response += f"These projects span {len(locations)} different locations. "
        
        # Add cost information if available
        total_cost = sum(float(r.get('ContractCost', 0)) for r in records if r.get('ContractCost'))
        if total_cost > 0:
            response += f"The total investment across all projects is approximately ₱{total_cost:,.2f}. "
            
            avg_cost = total_cost / len(records)
            response += f"The average project cost is around ₱{avg_cost:,.2f}.\n\n"
        
        # Add project highlights
        response += "Here are some notable projects:\n\n"
        
        for i, record in enumerate(records[:3], 1):
            project_desc = record.get('ProjectDescription', 'A flood control project')
            location = f"{record.get('Municipality', 'an unspecified location')}, {record.get('Province', 'an unspecified province')}"
            cost = float(record.get('ContractCost', 0))
            contractor = record.get('Contractor')
            year = record.get('CompletionYear')
            
            response += f"{i}. {project_desc} in {location} "
            response += f"(₱{cost:,.2f}) "
            
            if contractor and contractor.lower() != 'n/a':
                response += f"by {contractor} "
            if year and str(year).strip() and str(year).lower() != 'n/a':
                response += f"(Completed: {year})"
                
            response += "\n"
        
        # Add a note about additional results
        if len(records) > 3:
            response += f"\nThere are {len(records) - 3} more projects available. "
            response += "Would you like me to provide more details about any specific project?"
        
        return response
    
    def _format_metadata_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a natural language overview of the dataset's metadata."""
        if not records:
            return "I couldn't retrieve any metadata about the dataset at the moment."
        
        metadata = records[0]
        response = "Here's what I can tell you about the flood control projects dataset:\n\n"
        
        # Basic dataset statistics
        total_projects = metadata.get('total_projects', 0)
        total_investment = float(metadata.get('total_investment', 0))
        
        response += f"The dataset contains information on {total_projects:,} flood control projects "
        
        # Add time coverage if available
        if 'earliest_year' in metadata and 'latest_year' in metadata:
            response += f"carried out between {metadata['earliest_year']} and {metadata['latest_year']}. "
        
        if total_investment > 0:
            response += f"The total investment across all projects is approximately ₱{total_investment:,.2f}. "
            
            # Add average project cost
            if total_projects > 0:
                avg_cost = total_investment / total_projects
                response += f"On average, each project costs around ₱{avg_cost:,.2f}.\n\n"
        
        # Add infrastructure types if available
        if 'unique_infra_types' in metadata and metadata['unique_infra_types']:
            infra_types = metadata['unique_infra_types']
            response += "\n**Types of Infrastructure:**\n"
            for i, infra_type in enumerate(infra_types[:10], 1):
                response += f"- {infra_type}"
                if i < len(infra_types[:10]):
                    response += "\n"
        
        # Add contractors information if available
        if 'unique_contractors_list' in metadata and metadata['unique_contractors_list']:
            contractors = metadata['unique_contractors_list']
            response += f"\n\nThe dataset includes {len(contractors)} different contractors. "
            
            if len(contractors) > 10:
                response += f"Here are some of the main contractors: {', '.join(contractors[:5])}."
                if len(contractors) > 5:
                    response += f" There are {len(contractors) - 5} additional contractors in the dataset."
        
        # Add data quality information if available
        if 'missing_data_summary' in metadata and metadata['missing_data_summary']:
            missing_data = metadata['missing_data_summary']
            if any(count > 0 for count in missing_data.values()):
                response += "\n\n**Data Quality Notes:**\n"
                response += "Some data points are missing in the following fields:\n"
                for column, missing_count in missing_data.items():
                    if missing_count > 0:
                        percentage = (missing_count / metadata.get('total_projects', 1)) * 100
                        response += f"- {column}: {missing_count} projects ({percentage:.1f}%) missing data\n"
        
        # Add a helpful closing
        response += "\nWhat specific aspect of the dataset would you like to explore further?"
        
        return response
    
    def _extract_infrastructure_types(self, description: str) -> List[str]:
        """Extract infrastructure types from project descriptions."""
        if not description:
            return []
            
        # Common infrastructure types to look for
        infrastructure_terms = [
            'drainage', 'flood wall', 'seawall', 'dike', 'levee', 'floodgate', 
            'pump station', 'retention basin', 'detention pond', 'channel',
            'culvert', 'bridge', 'embankment', 'floodway', 'spillway', 'canal',
            'tunnel', 'flood control', 'river control', 'shore protection'
        ]
        
        # Find matching infrastructure types in the description
        found_types = []
        description_lower = description.lower()
        
        for term in infrastructure_terms:
            if term in description_lower:
                # Add the properly capitalized version
                found_types.append(term.title())
                
        return found_types
        
    def _format_analysis_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a natural language response for analysis and insights queries."""
        if not records:
            return "I couldn't find any analysis results for your query. Could you try rephrasing or asking about something else?"
        
        query_lower = query.lower()
        
        # Check if this is an infrastructure type query
        infra_terms = ['type of infrastructure', 'infrastructure type', 'kind of project', 'what kind of', 'what types of']
        is_infra_query = any(term in query_lower for term in infra_terms)
        
        # Extract location if specified
        location = None
        location_terms = ['in ', 'at ', 'for ', 'near ']
        for term in location_terms:
            if term in query_lower:
                loc_start = query_lower.find(term) + len(term)
                location = query[loc_start:].split('.')[0].split('?')[0].strip()
                break
        
        # Handle infrastructure type queries
        if is_infra_query:
            # Filter by location if specified
            if location:
                filtered_records = []
                for r in records:
                    if (location.lower() in str(r.get('Municipality', '')).lower() or 
                        location.lower() in str(r.get('Province', '')).lower() or
                        location.lower() in str(r.get('Region', '')).lower()):
                        filtered_records.append(r)
                records = filtered_records
                
                if not records:
                    return f"I couldn't find any flood control projects in {location} to analyze infrastructure types."
            
            # Extract and count infrastructure types
            type_counts = {}
            total_projects = 0
            
            for record in records:
                desc = record.get('ProjectDescription', '')
                if not desc:
                    continue
                    
                types_in_project = self._extract_infrastructure_types(desc)
                if not types_in_project:
                    types_in_project = ['Other Flood Control']
                    
                for infra_type in types_in_project:
                    type_counts[infra_type] = type_counts.get(infra_type, 0) + 1
                    total_projects += 1
            
            if not type_counts:
                return "I couldn't determine the types of infrastructure from the available project data."
            
            # Sort by frequency
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Build response
            location_phrase = f" in {location}" if location else ""
            response = f"## 🏗️ Types of Flood Control Infrastructure{location_phrase}\n\n"
            
            if total_projects > 0:
                response += f"Based on {total_projects} analyzed projects{location_phrase.lower() if location else ''}, "
                response += "here are the most common types of flood control infrastructure:\n\n"
                
                for i, (infra_type, count) in enumerate(sorted_types[:10], 1):  # Limit to top 10
                    percentage = (count / total_projects) * 100
                    response += f"{i}. **{infra_type}**: {count} projects ({percentage:.1f}%)\n"
                
                if len(sorted_types) > 10:
                    response += f"\n*And {len(sorted_types) - 10} other types of infrastructure.*"
                
                response += "\n\n*Note: This analysis is based on project descriptions and may not capture all infrastructure types.*"
                
                return response
        
        # Check if this is a region count query
        is_region_query = any(term in query_lower for term in ["region", "regions", "which region"])
        is_count_query = any(term in query_lower for term in ["count", "number", "how many", "most"])
        
        # Special handling for region count queries
        if is_region_query and is_count_query:
            # Extract region data from records
            region_counts = {}
            for record in records:
                region = record.get('Region')
                if region and str(region).strip().lower() != 'n/a':
                    region = str(region).strip()
                    region_counts[region] = region_counts.get(region, 0) + 1
            
            if not region_counts:
                return "I couldn't find any region information in the data. The records might be missing region details."
            
            # Sort regions by project count
            sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
            total_projects = sum(region_counts.values())
            
            # Build response
            response = f"I analyzed {total_projects:,} flood control projects across different regions. Here's what I found:\n\n"
            
            # Top regions
            top_region, top_count = sorted_regions[0]
            response += f"🏆 **{top_region}** has the most projects with **{top_count}** "
            response += f"({(top_count/total_projects*100):.1f}% of all projects).\n\n"
            
            # Show top 5 regions
            response += "**Top Regions by Number of Projects:**\n"
            for i, (region, count) in enumerate(sorted_regions[:5], 1):
                percentage = (count / total_projects) * 100
                response += f"{i}. **{region}**: {count} projects ({percentage:.1f}%)\n"
            
            # Add comparison with bottom regions if significant
            if len(sorted_regions) > 5:
                bottom_region, bottom_count = sorted_regions[-1]
                if bottom_count < top_count:  # Only show if there's a difference
                    response += f"\nIn contrast, {bottom_region} has the fewest projects with just {bottom_count}."
            
            # Add note about total regions
            response += f"\n*Projects are distributed across {len(region_counts)} regions in total.*"
            
            return response
            
        # Standard analysis response for other queries
        analysis = records[0] if records else {}
        response = "Here's what I found based on your query:\n\n"
        
        # Distribution analysis
        if 'region_distribution' in analysis and analysis['region_distribution']:
            region_dist = analysis['region_distribution']
            total_projects = sum(region_dist.values())
            
            response += f"I've analyzed {total_projects:,} projects across different regions. "
            
            # Sort regions by number of projects
            sorted_regions = sorted(region_dist.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_regions) > 0:
                top_region, top_count = sorted_regions[0]
                response += f"The region with the most projects is **{top_region}** with **{top_count}** projects. "
                
                if len(sorted_regions) > 1:
                    bottom_region, bottom_count = sorted_regions[-1]
                    if bottom_count < top_count:  # Only show if there's a difference
                        response += f"In contrast, {bottom_region} has the fewest projects with just {bottom_count}. "
            
            response += "\n\n**Project Distribution by Region:**\n"
            
            # Show top 5 regions
            for i, (region, count) in enumerate(sorted_regions[:5], 1):
                percentage = (count / total_projects) * 100
                response += f"- {region}: {count} projects ({percentage:.1f}% of total)\n"
            
            if len(sorted_regions) > 5:
                response += f"- ... and {len(sorted_regions) - 5} other regions\n"
            
            response += "\n"  # Add spacing before next section
        if 'cost_analysis' in analysis and analysis['cost_analysis']:
            cost_data = analysis['cost_analysis']
            
            response += "**Cost Analysis**\n"
            
            if 'average_cost' in cost_data:
                response += f"- The average project costs approximately ₱{float(cost_data['average_cost']):,.2f}. "
                
                if 'min_cost' in cost_data and 'max_cost' in cost_data:
                    response += f"Project costs range from ₱{float(cost_data['min_cost']):,.2f} to ₱{float(cost_data['max_cost']):,.2f}. "
                    
                    # Add context about the range
                    cost_ratio = float(cost_data['max_cost']) / float(cost_data['min_cost']) if float(cost_data['min_cost']) > 0 else 0
                    if cost_ratio > 100:
                        response += "There's significant variation in project costs. "
                    elif cost_ratio > 10:
                        response += "Project costs vary considerably. "
                    
                response += "\n"
            
            if 'total_investment' in cost_data:
                response += f"- Total investment across all projects is approximately ₱{float(cost_data['total_investment']):,.2f}.\n"
            
            response += "\n"
        
        # Completion time analysis
        if 'completion_analysis' in analysis and analysis['completion_analysis']:
            completion_data = analysis['completion_analysis']
            response += "**Project Completion Times**\n"
            
            if 'average_duration_days' in completion_data:
                avg_days = float(completion_data['average_duration_days'])
                avg_months = avg_days / 30.44
                avg_years = avg_days / 365.25
                
                if avg_days < 30:
                    duration_str = f"{avg_days:.1f} days"
                elif avg_days < 365:
                    duration_str = f"{avg_months:.1f} months"
                else:
                    duration_str = f"{avg_years:.1f} years"
                
                response += f"- Projects take {duration_str} to complete on average. "
                
                if 'min_duration_days' in completion_data and 'max_duration_days' in completion_data:
                    min_days = float(completion_data['min_duration_days'])
                    max_days = float(completion_data['max_duration_days'])
                    
                    # Format min duration
                    if min_days < 30:
                        min_str = f"{min_days:.0f} days"
                    elif min_days < 365:
                        min_str = f"{min_days/30.44:.1f} months"
                    else:
                        min_str = f"{min_days/365.25:.1f} years"
                    
                    # Format max duration
                    if max_days < 30:
                        max_str = f"{max_days:.0f} days"
                    elif max_days < 365:
                        max_str = f"{max_days/30.44:.1f} months"
                    else:
                        max_str = f"{max_days/365.25:.1f} years"
                    
                    response += f"Completion times range from {min_str} to {max_str}. "
                    
                    # Add context about the range
                    if max_days / min_days > 10 and min_days > 0:
                        response += "This wide range suggests that project complexity varies significantly. "
                
                response += "\n\n"
        
        # Province distribution
        if 'province_distribution' in analysis and analysis['province_distribution']:
            province_dist = analysis['province_distribution']
            sorted_provinces = sorted(province_dist.items(), key=lambda x: x[1], reverse=True)
            
            response += "**Project Distribution by Province**\n"
            response += f"Projects are distributed across {len(sorted_provinces)} provinces. "
            
            if len(sorted_provinces) >= 3:
                top_provinces = ", ".join([f"{p[0]} ({p[1]} projects)" for p in sorted_provinces[:3]])
                response += f"The top provinces are {top_provinces}. "
                
                # Add context about the distribution
                top_3_total = sum(p[1] for p in sorted_provinces[:3])
                percentage = (top_3_total / sum(province_dist.values())) * 100
                response += f"These account for {percentage:.1f}% of all projects. "
            
            response += "\n\n"
        
        # Yearly trends
        if 'year_distribution' in analysis and analysis['year_distribution']:
            year_dist = analysis['year_distribution']
            sorted_years = sorted(year_dist.items(), key=lambda x: x[0])
            
            # Only proceed if we have meaningful year data
            if len(sorted_years) > 1:
                response += "**Yearly Project Trends**\n"
                
                # Find the year with most projects
                max_year, max_count = max(year_dist.items(), key=lambda x: x[1])
                min_year, min_count = min(year_dist.items(), key=lambda x: x[1])
                
                response += f"- The year with the most projects was {max_year} with {max_count} projects. "
                
                # Calculate year-over-year changes
                yearly_changes = []
                for i in range(1, len(sorted_years)):
                    prev_year, prev_count = sorted_years[i-1]
                    curr_year, curr_count = sorted_years[i]
                    change = ((curr_count - prev_count) / prev_count) * 100 if prev_count > 0 else 0
                    yearly_changes.append((f"{prev_year}-{curr_year}", change))
                
                # Add trend information if we have enough data
                if len(yearly_changes) > 0:
                    avg_change = sum(c[1] for c in yearly_changes) / len(yearly_changes)
                    if abs(avg_change) > 5:  # Only mention if there's a significant trend
                        trend = "increasing" if avg_change > 0 else "decreasing"
                        response += f"On average, the number of projects has been {trend} by {abs(avg_change):.1f}% annually. "
                
                response += "\n\n"
        
        # Add a closing note
        response += "Would you like me to explore any specific aspect of this analysis in more detail?"
        
        return response
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generate helpful response when no results are found."""
        return f"""## No Results Found

I couldn't find any flood control projects matching your query: "{query}"

### 💡 Try These Suggestions:
- **Check spelling** of location names (e.g., "Cebu City", "Manila", "Davao")
- **Use broader terms** (e.g., "drainage projects" instead of specific technical terms)
- **Try different keywords** (e.g., "expensive projects", "recent completions")
- **Include region names** (e.g., "Region VII", "NCR", "Mindanao")

### 📋 Popular Search Examples:
- "Most expensive projects in Cebu"
- "Drainage projects completed in 2023"
- "Top contractors in Region VII"
- "Bridge projects in Manila"
- "Recent flood control projects"

*The database contains 9,800+ flood control projects across all Philippine regions.*
"""
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        if not self.client:
            logger.warning("OpenAI client is not initialized")
            return False
            
        try:
            # Test the client with a simple request
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI client test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "provider": "OpenAI",
            "available": str(self.is_available())
        }
