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
                    # st.success("✅ API key found in Streamlit secrets")  # Commented out to hide message
                else:
                    # st.warning("⚠️ OPENAI_API_KEY not found in Streamlit secrets")  # Commented out
                    # Show available keys for debugging
                    available_keys = list(st.secrets.keys()) if st.secrets else []
                    if available_keys:
                        pass  # st.info(f"Available secret keys: {available_keys}")  # Commented out
            else:
                pass  # st.warning("⚠️ Streamlit secrets not available")  # Commented out
        except Exception as e:
            pass  # st.error(f"Error accessing Streamlit secrets: {str(e)}")  # Commented out
        
        # Fallback to environment variables if no API key found
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                pass  # st.info("✅ API key found in environment variables")  # Commented out
        
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
                        # st.success(f"✅ OpenAI client initialized successfully using {strategy_name}")  # Commented out
                        return
                except Exception as e:
                    # st.warning(f"❌ {strategy_name} failed: {str(e)}")  # Commented out
                    continue
            
            # If all strategies fail
            # st.error("⚠️ All OpenAI initialization strategies failed")  # Commented out
            # st.info("The app will continue with basic functionality (no AI responses)")  # Commented out
            self.client = None
        else:
            pass  # st.error("⚠️ No OpenAI API key found in secrets or environment")  # Commented out
    
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
    
    def _get_system_prompt(self, query_type: str) -> str:
        """Get the appropriate system prompt based on query type."""
        prompts = {
            'location': 'You are an expert analyst of Philippine flood control projects. Provide detailed location-based information about flood control projects, including specific regions, provinces, cities, and municipalities.',
            'time': 'You are a time-series analyst for flood control projects. Provide detailed temporal information about project timelines, including start dates, completion dates, and durations.',
            'cost': 'You are a financial analyst specializing in infrastructure projects. Provide detailed cost information, including budgets, contract amounts, and cost comparisons.',
            'contractor': 'You are a construction industry expert. Provide detailed information about contractors, including their project history and performance metrics.',
            'type': 'You are a civil engineering expert. Provide detailed information about different types of flood control projects and their specifications.',
            'general': 'You are a helpful assistant that provides information about flood control projects in the Philippines. Answer questions clearly and concisely.'
        }
        return prompts.get(query_type, prompts['general'])

    def _prepare_prompt(self, query: str, records: List[Dict[str, Any]], context: Dict[str, Any] = None, query_type: str = 'general') -> str:
        """Prepare the prompt for the LLM based on the query and records."""
        # Format the records into a readable string
        records_str = "\n\n".join([
            f"Record {i+1}:\n" + "\n".join([f"{k}: {v}" for k, v in record.items()])
            for i, record in enumerate(records)
        ])
        
        # Include context if available
        context_str = ""
        if context:
            context_str = f"\n\nContext about the dataset:\n{context}"
        
        return f"""Question: {query}

Relevant project records:
{records_str}{context_str}

Please provide a clear and concise answer to the question based on the project records above. If the information is not available in the records, please state that clearly."""

    def generate_response(self, query: str, records: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """
        Generate a response based on the query and relevant records.
        
        Args:
            query: User's question
            records: Relevant records from the database
            context: Additional context about the dataset
            
        Returns:
            Formatted response string
        """
        if not records:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing your query or ask about something else."
            
        try:
            # Determine query type
            query_type = self._classify_query(query)
            
            # If no client is available, use a fallback response
            if not self.client:
                return self._fallback_response(query, records)
            
            # Prepare the prompt with context and records
            prompt = self._prepare_prompt(query, records, context, query_type)
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(query_type)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=1000
            )
            
            # Extract and clean the response
            response_text = response.choices[0].message.content.strip()
            return response_text
            
        except Exception as e:
            # If there's an error with the LLM, fall back to a simpler response
            print(f"Error generating response: {str(e)}")
            return self._fallback_response(query, records)
            
    def _fallback_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when the LLM is not available."""
        if not records:
            return "I couldn't find any information matching your query. Please try rephrasing your question."
            
        # Try to generate a basic response based on the records
        try:
            # Count records by contractor if it's a contractor query
            if any(term in query.lower() for term in ['contractor', 'company']):
                contractors = {}
                for record in records:
                    contractor = record.get('contractor', 'Unknown')
                    contractors[contractor] = contractors.get(contractor, 0) + 1
                
                response = ["Here are the contractors I found:"]
                for i, (contractor, count) in enumerate(sorted(contractors.items(), key=lambda x: x[1], reverse=True)[:5], 1):
                    response.append(f"{i}. {contractor} ({count} projects)")
                return "\n".join(response)
                
            # For location queries, list unique locations
            elif any(term in query.lower() for term in ['location', 'region', 'province', 'city']):
                locations = {}
                for record in records:
                    location = record.get('location', 'Unknown')
                    locations[location] = locations.get(location, 0) + 1
                
                response = ["Here are the locations I found:"]
                for i, (location, count) in enumerate(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5], 1):
                    response.append(f"{i}. {location} ({count} projects)")
                return "\n".join(response)
                
            # For cost queries, show the most expensive projects
            elif any(term in query.lower() for term in ['cost', 'price', 'expensive', 'cheap']):
                sorted_records = sorted(
                    [r for r in records if 'contract_cost' in r],
                    key=lambda x: float(str(x.get('contract_cost', 0).replace(',', '').replace('₱', '') or 0)),
                    reverse=True
                )[:5]
                
                response = ["Here are the most expensive projects I found:"]
                for i, record in enumerate(sorted_records, 1):
                    name = record.get('project_name', 'Unnamed Project')
                    cost = record.get('contract_cost', 'N/A')
                    response.append(f"{i}. {name} - {cost}")
                return "\n".join(response)
                
            # Default response
            return f"I found {len(records)} relevant projects. Here are some details:\n\n" + \
                   "\n\n".join([f"• {r.get('project_name', 'Unnamed Project')} - {r.get('contractor', 'Unknown')}" for r in records[:5]])
                   
        except Exception as e:
            print(f"Error in fallback response: {str(e)}")
            return "I found some information but couldn't process it properly. Please try rephrasing your question."
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of question being asked."""
        query_lower = query.lower()
        
        # Location-based questions
        location_terms = ['region', 'province', 'city', 'municipality', 'barangay', 'in ', 'at ', 'from ']
        if any(term in query_lower for term in location_terms):
            return 'location'
            
        # Time-based questions
        time_terms = ['year', 'month', 'date', 'day', 'week', 'when', 'complet', 'start', 'end', 'duration']
        if any(term in query_lower for term in time_terms):
            return 'time'
            
        # Cost/budget questions
        cost_terms = ['cost', 'price', 'budget', 'expensive', 'cheap', 'amount', 'fund', 'investment']
        if any(term in query_lower for term in cost_terms):
            return 'cost'
            
        # Contractor questions
        contractor_terms = ['contractor', 'company', 'implement', 'award', 'won', 'bid']
        if any(term in query_lower for term in contractor_terms):
            return 'contractor'
            
        # Project type questions
        type_terms = ['type', 'kind', 'category', 'classification']
        if any(term in query_lower for term in type_terms):
            return 'type'
            
        # Default to general information
        return 'general'
    
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
        """Generate a comprehensive fallback response when LLM is not available."""
        if not relevant_records:
            return self._generate_no_results_response(query)
        
        # Analyze query type for better response formatting
        query_type = self._analyze_query_intent(query.lower())
        
        if query_type == 'cost_analysis':
            return self._format_cost_response(query, relevant_records)
        elif query_type == 'contractor_analysis':
            return self._format_contractor_response(query, relevant_records)
        elif query_type == 'completion_analysis':
            return self._format_completion_response(query, relevant_records)
        elif query_type == 'project_type_analysis':
            return self._format_project_type_response(query, relevant_records)
        elif query_type == 'location_analysis':
            return self._format_location_response(query, relevant_records)
        elif query_type == 'comparison':
            return self._format_comparison_response(query, relevant_records)
        elif query_type == 'metadata_analysis':
            return self._format_metadata_response(query, relevant_records)
        elif query_type == 'analysis_insights':
            return self._format_analysis_response(query, relevant_records)
        else:
            return self._format_general_response(query, relevant_records)
    
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze query to determine the best response format."""
        if any(term in query for term in ['expensive', 'cost', 'budget', 'price', 'amount']):
            return 'cost_analysis'
        elif any(term in query for term in ['contractor', 'company', 'builder', 'who built']):
            return 'contractor_analysis'
        elif any(term in query for term in ['completed', 'finished', 'when', 'completion']):
            return 'completion_analysis'
        elif any(term in query for term in ['type', 'drainage', 'bridge', 'seawall', 'revetment']):
            return 'project_type_analysis'
        elif any(term in query for term in ['where', 'location', 'region', 'province', 'city']):
            return 'location_analysis'
        else:
            return 'general'
    
    def _format_cost_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for cost-related queries."""
        response_parts = []
        
        # Check if this is a cost summary query
        if records and records[0].get('query_type') == 'cost_summary':
            return self._format_cost_summary_response(query, records)
        
        if any(term in query.lower() for term in ['cheapest', 'lowest', 'least', 'smallest', 'minimum', 'low budget', 'low cost']):
            response_parts.append("# 💰 Most Affordable Flood Control Projects")
        elif any(term in query.lower() for term in ['expensive', 'highest', 'most', 'largest', 'maximum', 'high budget', 'high cost']):
            response_parts.append("# 💰 Most Expensive Flood Control Projects")
        else:
            response_parts.append("# 💰 Flood Control Projects by Cost")
        
        response_parts.append("")
        
        # Calculate total cost
        total_cost = sum(float(r.get('ContractCost', 0)) for r in records if r.get('ContractCost'))
        if total_cost > 0:
            response_parts.append(f"**📊 Total Combined Investment:** ₱{total_cost:,.2f}")
            response_parts.append("")
        
        for i, record in enumerate(records[:5], 1):
            cost = record.get('ContractCost')
            abc = record.get('ABC')
            
            response_parts.append(f"## {i}. {record.get('ProjectDescription', 'Unknown Project')}")
            response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
            
            if cost:
                response_parts.append(f"**💰 Contract Cost:** ₱{float(cost):,.2f}")
            if abc and abc != cost:
                response_parts.append(f"**📋 Approved Budget:** ₱{float(abc):,.2f}")
                if cost and abc:
                    savings = float(abc) - float(cost)
                    if savings > 0:
                        response_parts.append(f"**💡 Budget Savings:** ₱{savings:,.2f}")
            
            response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
            response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
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
        """Format response for contractor-related queries."""
        response_parts = ["# 🏗️ Contractor Analysis", ""]
        
        # Check if we have contractor project counts
        if records and 'contractor_project_count' in records[0]:
            response_parts.append("**📊 Top Contractors by Project Count**")
            response_parts.append("")
            
            for i, record in enumerate(records[:5], 1):
                contractor = record.get('Contractor', 'Unknown')
                count = record.get('contractor_project_count', 0)
                cost = record.get('ContractCost', 0)
                
                response_parts.append(f"## {i}. {contractor}")
                response_parts.append(f"**📊 Total Projects:** {count}")
                response_parts.append(f"**💰 Sample Project Cost:** ₱{float(cost):,.2f}")
                response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
                response_parts.append(f"**🏗️ Sample Project:** {record.get('ProjectDescription', 'N/A')}")
                response_parts.append("")
        else:
            # Regular contractor search results
            for i, record in enumerate(records[:5], 1):
                response_parts.append(f"## {i}. {record.get('Contractor', 'Unknown Contractor')}")
                response_parts.append(f"**🏗️ Project:** {record.get('ProjectDescription', 'N/A')}")
                response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
                response_parts.append(f"**💰 Contract Cost:** ₱{float(record.get('ContractCost', 0)):,.2f}")
                response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
                response_parts.append("")
        
        return "\n".join(response_parts)
    
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
        """Format standard location response."""
        response_parts = ["# 📍 Location-Based Project Analysis", ""]
        
        # Get unique locations
        locations = set(f"{r.get('Municipality', 'N/A')}, {r.get('Province', 'N/A')}" for r in records)
        if len(locations) > 1:
            response_parts.append(f"**📊 Locations Covered:** {len(locations)} municipalities/cities")
            response_parts.append("")
        
        # Calculate total investment in the area
        total_cost = sum(float(r.get('ContractCost', 0)) for r in records if r.get('ContractCost'))
        if total_cost > 0:
            response_parts.append(f"**💰 Total Investment:** ₱{total_cost:,.2f}")
            response_parts.append("")
        
        for i, record in enumerate(records[:5], 1):
            response_parts.append(f"## {i}. {record.get('ProjectDescription', 'Unknown Project')}")
            response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
            response_parts.append(f"**🌏 Region:** {record.get('Region', 'N/A')}")
            response_parts.append(f"**🔧 Type:** {record.get('TypeofWork', 'N/A')}")
            response_parts.append(f"**💰 Contract Cost:** ₱{float(record.get('ContractCost', 0)):,.2f}")
            response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
            response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _format_comparison_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for comparison queries."""
        response_parts = ["# ⚖️ Comparison Analysis", ""]
        
        # Check if records have comparison metadata
        if records and 'comparison_term' in records[0]:
            # Specific entity comparison
            response_parts.append("**🔍 Side-by-Side Comparison**")
            response_parts.append("")
            
            for i, record in enumerate(records[:5], 1):
                term = record.get('comparison_term', f'Entity {i}')
                response_parts.append(f"## {term.title()}")
                response_parts.append(f"**🏗️ Top Project:** {record.get('ProjectDescription', 'N/A')}")
                response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
                response_parts.append(f"**💰 Contract Cost:** ₱{float(record.get('ContractCost', 0)):,.2f}")
                response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
                response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
                response_parts.append("")
        
        elif records and 'comparison_category' in records[0]:
            # Category-based comparison
            category = records[0].get('comparison_category', 'category')
            response_parts.append(f"**📊 Top Projects by {category.title()}**")
            response_parts.append("")
            
            for i, record in enumerate(records[:5], 1):
                category_value = record.get('Region', record.get('Province', record.get('Municipality', f'Item {i}')))
                response_parts.append(f"## {i}. {category_value}")
                response_parts.append(f"**🏗️ Project:** {record.get('ProjectDescription', 'N/A')}")
                response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
                response_parts.append(f"**💰 Contract Cost:** ₱{float(record.get('ContractCost', 0)):,.2f}")
                response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
                response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
                response_parts.append("")
        
        else:
            # General comparison
            response_parts.append("**🏆 Top Projects for Comparison**")
            response_parts.append("")
            
            # Calculate some comparison metrics
            total_cost = sum(float(r.get('ContractCost', 0)) for r in records if r.get('ContractCost'))
            avg_cost = total_cost / len(records) if records else 0
            
            response_parts.append(f"**💰 Total Value:** ₱{total_cost:,.2f}")
            response_parts.append(f"**📊 Average Cost:** ₱{avg_cost:,.2f}")
            response_parts.append("")
            
            for i, record in enumerate(records[:5], 1):
                cost = float(record.get('ContractCost', 0))
                cost_vs_avg = ((cost - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
                
                response_parts.append(f"## {i}. {record.get('ProjectDescription', 'Unknown Project')}")
                response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
                response_parts.append(f"**💰 Contract Cost:** ₱{cost:,.2f} ({cost_vs_avg:+.1f}% vs average)")
                response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
                response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
                response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _format_general_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format general response for other queries."""
        response_parts = [
            f"# 🔍 Search Results",
            f"**📊 Found {len(records)} flood control projects**",
            ""
        ]
        
        for i, record in enumerate(records[:5], 1):
            response_parts.append(f"## {i}. {record.get('ProjectDescription', 'Unknown Project')}")
            response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
            response_parts.append(f"**🔧 Type:** {record.get('TypeofWork', 'N/A')}")
            response_parts.append(f"**💰 Contract Cost:** ₱{float(record.get('ContractCost', 0)):,.2f}")
            response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
            response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
            response_parts.append("")
        
        if len(records) > 5:
            response_parts.append(f"*Showing top 5 results out of {len(records)} found.*")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _format_metadata_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for metadata and system queries."""
        if not records:
            return "No metadata information available."
        
        metadata = records[0]
        response_parts = ["# 📊 Dataset Information", ""]
        
        # Basic dataset statistics
        if 'total_projects' in metadata:
            response_parts.append("## 📈 Dataset Overview")
            response_parts.append(f"**📊 Total Projects:** {metadata['total_projects']:,}")
            response_parts.append(f"**📋 Total Columns:** {metadata.get('total_columns', 'N/A')}")
            response_parts.append("")
        
        # Geographic coverage
        if any(key in metadata for key in ['unique_regions', 'unique_provinces', 'unique_municipalities']):
            response_parts.append("## 🌍 Geographic Coverage")
            if 'unique_regions' in metadata:
                response_parts.append(f"**🏛️ Regions:** {metadata['unique_regions']}")
            if 'unique_provinces' in metadata:
                response_parts.append(f"**🏙️ Provinces:** {metadata['unique_provinces']}")
            if 'unique_municipalities' in metadata:
                response_parts.append(f"**🏘️ Municipalities:** {metadata['unique_municipalities']}")
            response_parts.append("")
        
        # Contractor information
        if 'unique_contractors' in metadata:
            response_parts.append("## 🏗️ Contractor Information")
            response_parts.append(f"**👷 Unique Contractors:** {metadata['unique_contractors']}")
            response_parts.append("")
        
        # Date range
        if 'date_range_start' in metadata and 'date_range_end' in metadata:
            response_parts.append("## 📅 Timeline Coverage")
            start_year = metadata['date_range_start']
            end_year = metadata['date_range_end']
            if start_year and end_year:
                response_parts.append(f"**📅 Year Range:** {start_year} - {end_year}")
                response_parts.append(f"**⏱️ Coverage Period:** {end_year - start_year + 1} years")
            response_parts.append("")
        
        # Specific query responses
        if 'unique_infra_types' in metadata:
            response_parts.append("## 🔧 Infrastructure Types")
            infra_types = metadata['unique_infra_types']
            if infra_types:
                for i, infra_type in enumerate(infra_types[:10], 1):
                    response_parts.append(f"{i}. {infra_type}")
            response_parts.append("")
        
        if 'unique_contractors_list' in metadata:
            response_parts.append("## 🏗️ All Contractors")
            contractors = metadata['unique_contractors_list']
            if contractors:
                response_parts.append(f"**Total:** {len(contractors)} contractors")
                response_parts.append("**Sample contractors:**")
                for i, contractor in enumerate(contractors[:10], 1):
                    response_parts.append(f"{i}. {contractor}")
                if len(contractors) > 10:
                    response_parts.append(f"*... and {len(contractors) - 10} more contractors*")
            response_parts.append("")
        
        # Data quality information
        if 'missing_data_summary' in metadata:
            missing_data = metadata['missing_data_summary']
            if any(count > 0 for count in missing_data.values()):
                response_parts.append("## ⚠️ Data Quality")
                response_parts.append("**Columns with missing data:**")
                for column, missing_count in missing_data.items():
                    if missing_count > 0:
                        percentage = (missing_count / metadata.get('total_projects', 1)) * 100
    
    def _format_analysis_response(self, query: str, records: List[Dict[str, Any]]) -> str:
        """Format response for analysis and insights queries."""
        if not records:
            return "No analysis data available."
        
        analysis = records[0]
        response_parts = ["# 📊 Analysis & Insights", ""]
        
        # Distribution analysis
        if 'region_distribution' in analysis:
            response_parts.append("## 🌍 Project Distribution by Region")
            region_dist = analysis['region_distribution']
            total_projects = sum(region_dist.values())
            
            response_parts.append(f"**📊 Total Projects Analyzed:** {total_projects:,}")
            response_parts.append("")
            
            # Sort regions by project count
            sorted_regions = sorted(region_dist.items(), key=lambda x: x[1], reverse=True)
            
            for i, (region, count) in enumerate(sorted_regions[:10], 1):
                percentage = (count / total_projects) * 100
                response_parts.append(f"**{i}. {region}**")
                response_parts.append(f"   📊 Projects: {count:,} ({percentage:.1f}%)")
                response_parts.append("")
        
        if 'province_distribution' in analysis:
            response_parts.append("## 🏙️ Top Provinces by Project Count")
            province_dist = analysis['province_distribution']
            
            for i, (province, count) in enumerate(province_dist.items(), 1):
                response_parts.append(f"**{i}. {province}:** {count:,} projects")
            response_parts.append("")
        
        if 'year_distribution' in analysis:
            response_parts.append("## 📅 Projects by Completion Year")
            year_dist = analysis['year_distribution']
            
            # Sort by year
            sorted_years = sorted(year_dist.items())
            
            for year, count in sorted_years:
                if year and str(year) != 'nan':
                    response_parts.append(f"**{int(year)}:** {count:,} projects")
            response_parts.append("")
        
        # Funding trend analysis
        if 'funding_trend' in analysis:
            response_parts.append("## 💰 Funding Trend Analysis")
            funding_trend = analysis['funding_trend']
            
            # Sort by year
            sorted_funding = sorted(funding_trend.items())
            total_funding = sum(funding_trend.values())
            
            response_parts.append(f"**💰 Total Investment:** ₱{total_funding:,.2f}")
            response_parts.append("")
            
            response_parts.append("**📈 Year-by-Year Breakdown:**")
            for year, amount in sorted_funding:
                if year and str(year) != 'nan':
                    percentage = (amount / total_funding) * 100 if total_funding > 0 else 0
                    response_parts.append(f"• **{int(year)}:** ₱{amount:,.2f} ({percentage:.1f}%)")
            response_parts.append("")
        
        # Top analysis
        if 'top_expensive_projects' in analysis:
            response_parts.append("## 💰 Most Expensive Projects")
            top_projects = analysis['top_expensive_projects']
            
            for i, project in enumerate(top_projects, 1):
                cost = project.get('ContractCost', 0)
                description = project.get('ProjectDescription', 'Unknown Project')
                location = f"{project.get('Municipality', 'N/A')}, {project.get('Province', 'N/A')}"
                
                response_parts.append(f"### {i}. {description}")
                response_parts.append(f"**📍 Location:** {location}")
                response_parts.append(f"**💰 Cost:** ₱{cost:,.2f}")
                response_parts.append("")
        
        if 'top_contractors' in analysis:
            response_parts.append("## 🏗️ Top Contractors by Project Count")
            top_contractors = analysis['top_contractors']
            
            for i, (contractor, count) in enumerate(top_contractors.items(), 1):
                response_parts.append(f"**{i}. {contractor}:** {count} projects")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generate helpful response when no results are found."""
        return f"""## ❌ No Results Found

I couldn't find any flood control projects matching your query: **"{query}"**

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
        return self.client is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "provider": "OpenAI",
            "available": str(self.is_available())
        }
