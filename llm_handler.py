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
        
        if 'expensive' in query.lower():
            response_parts.append("# 💰 Most Expensive Flood Control Projects")
        elif 'cheapest' in query.lower():
            response_parts.append("# 💰 Most Affordable Flood Control Projects")
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
        """Format response for project type queries."""
        response_parts = ["# 🔧 Project Type Analysis", ""]
        
        # Get unique project types
        project_types = set(r.get('TypeofWork', 'N/A') for r in records)
        if len(project_types) > 1:
            response_parts.append(f"**📊 Project Types Found:** {', '.join(project_types)}")
            response_parts.append("")
        
        for i, record in enumerate(records[:5], 1):
            response_parts.append(f"## {i}. {record.get('ProjectDescription', 'Unknown Project')}")
            response_parts.append(f"**🔧 Type of Work:** {record.get('TypeofWork', 'N/A')}")
            response_parts.append(f"**🏗️ Infrastructure Type:** {record.get('infra_type', 'N/A')}")
            response_parts.append(f"**📍 Location:** {record.get('Municipality', 'N/A')}, {record.get('Province', 'N/A')}")
            response_parts.append(f"**💰 Contract Cost:** ₱{float(record.get('ContractCost', 0)):,.2f}")
            response_parts.append(f"**🏗️ Contractor:** {record.get('Contractor', 'N/A')}")
            response_parts.append(f"**📅 Completion Year:** {record.get('CompletionYear', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
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
