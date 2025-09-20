import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class FloodControlDataHandler:
    """Handles CSV data loading, processing, and querying for flood control projects."""
    
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.text_columns = []
        
    def load_csv(self, uploaded_file) -> bool:
        """Load and validate CSV file from uploaded file."""
        try:
            self.df = pd.read_csv(uploaded_file)
            return self._process_loaded_data("uploaded CSV file")
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return False
    
    def load_csv_from_path(self, file_path: str) -> bool:
        """Load and validate CSV file from file path."""
        try:
            self.df = pd.read_csv(file_path)
            return self._process_loaded_data(f"dataset: {file_path}")
            
        except Exception as e:
            st.error(f"Error loading CSV file from {file_path}: {str(e)}")
            return False
    
    def _process_loaded_data(self, source_description: str) -> bool:
        """Process loaded data regardless of source."""
        try:
            # Basic validation
            if self.df.empty:
                st.error(f"The {source_description} is empty.")
                return False
                
            # Identify text columns for search
            self.text_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            
            # Create searchable text by combining all text columns
            self._prepare_search_index()
            
            return True
            
        except Exception as e:
            st.error(f"Error processing data from {source_description}: {str(e)}")
            return False
    
    def _prepare_search_index(self):
        """Prepare TF-IDF index for semantic search."""
        if self.df is None or self.df.empty:
            return
            
        # Combine all text columns into a single searchable text
        search_texts = []
        for _, row in self.df.iterrows():
            text_parts = []
            for col in self.text_columns:
                if pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            search_texts.append(" ".join(text_parts))
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(search_texts)
    
    def search_relevant_records(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for records most relevant to the query with intelligent processing."""
        if self.df is None or self.vectorizer is None:
            return []
        
        try:
            query_lower = query.lower()
            filtered_df = self.df.copy()
            
            # Extract location mentions
            location = self._extract_location(query_lower)
            if location:
                filtered_df = self._filter_by_location(filtered_df, location)
                # If location filtering returns no results, fall back to full dataset
                if filtered_df.empty:
                    filtered_df = self.df.copy()
            
            # Extract year mentions
            year = self._extract_year(query_lower)
            if year:
                year_filtered = self._filter_by_year(filtered_df, year)
                # Only apply year filter if it returns results
                if not year_filtered.empty:
                    filtered_df = year_filtered
            
            # Handle different query types
            query_type = self._classify_query(query_lower)
            
            if query_type == 'cost_analysis':
                return self._handle_cost_queries(filtered_df, query_lower, top_k)
            elif query_type == 'contractor_analysis':
                return self._handle_contractor_queries(filtered_df, query_lower, top_k)
            elif query_type == 'completion_analysis':
                return self._handle_completion_queries(filtered_df, query_lower, top_k)
            elif query_type == 'project_type_analysis':
                return self._handle_project_type_queries(filtered_df, query_lower, top_k)
            elif query_type == 'location_analysis':
                return self._handle_location_queries(filtered_df, query_lower, top_k)
            elif query_type == 'comparison':
                return self._handle_comparison_queries(filtered_df, query_lower, top_k)
            elif query_type == 'metadata_analysis':
                return self._handle_metadata_queries(filtered_df, query_lower, top_k)
            elif query_type == 'analysis_insights':
                return self._handle_analysis_queries(filtered_df, query_lower, top_k)
            else:
                # Default semantic search
                return self._semantic_search(filtered_df, query, top_k)
            
        except Exception as e:
            st.error(f"Error searching records: {str(e)}")
            return []
    
    def _extract_location(self, query: str) -> str:
        """Extract location mentions from query."""
        # Common Philippine locations - order matters (longer phrases first)
        locations = {
            'cebu city': ['cebu city'],
            'cebu': ['cebu'],
            'davao city': ['davao city'],
            'davao': ['davao'],
            'manila': ['manila', 'metro manila', 'ncr'],
            'palawan': ['palawan', 'puerto princesa'],
            'region i': ['region i', 'region 1', 'ilocos'],
            'region ii': ['region ii', 'region 2', 'cagayan'],
            'region iii': ['region iii', 'region 3', 'central luzon'],
            'region iv-a': ['region iv-a', 'region 4a', 'calabarzon'],
            'region iv-b': ['region iv-b', 'region 4b', 'mimaropa'],
            'region v': ['region v', 'region 5', 'bicol'],
            'region vi': ['region vi', 'region 6', 'western visayas'],
            'region vii': ['region vii', 'region 7', 'central visayas'],
            'region viii': ['region viii', 'region 8', 'eastern visayas'],
            'region ix': ['region ix', 'region 9', 'zamboanga'],
            'region x': ['region x', 'region 10', 'northern mindanao'],
            'region xi': ['region xi', 'region 11', 'davao region'],
            'region xii': ['region xii', 'region 12', 'region twelve', 'soccsksargen'],
            'region xiii': ['region xiii', 'region 13', 'caraga'],
            'barmm': ['barmm', 'bangsamoro', 'armm'],
            'car': ['car', 'cordillera']
        }
        
        # Sort by length of variants (longest first) to match "cebu city" before "cebu"
        sorted_locations = sorted(locations.items(), key=lambda x: max(len(v) for v in x[1]), reverse=True)
        
        for standard_name, variants in sorted_locations:
            for variant in variants:
                if variant in query:
                    return standard_name
        return None
    
    def _extract_year(self, query: str) -> int:
        """Extract year mentions from query."""
        import re
        years = re.findall(r'\b(20[0-9]{2})\b', query)
        return int(years[0]) if years else None
    
    def _filter_by_location(self, df: pd.DataFrame, location: str) -> pd.DataFrame:
        """Filter dataframe by location with flexible matching."""
        # Create multiple search patterns for better matching
        search_patterns = [location]
        
        # Add variations for common city names
        if location == 'cebu city':
            search_patterns.extend(['cebu city', 'cebu', 'capital.*cebu'])
        elif location == 'davao city':
            search_patterns.extend(['davao city', 'davao', 'capital.*davao'])
        elif location == 'manila':
            search_patterns.extend(['manila', 'metro manila', 'ncr'])
        elif location == 'region xii':
            search_patterns.extend(['region xii', 'region 12', 'soccsksargen'])
        
        # Create combined mask for all patterns
        location_mask = pd.Series([False] * len(df), index=df.index)
        
        for pattern in search_patterns:
            try:
                pattern_mask = (
                    df['Municipality'].str.contains(pattern, case=False, na=False, regex=True) |
                    df['Province'].str.contains(pattern, case=False, na=False, regex=True) |
                    df['Region'].str.contains(pattern, case=False, na=False, regex=True)
                )
                location_mask = location_mask | pattern_mask
            except Exception:
                # Skip problematic patterns
                continue
        
        return df[location_mask]
    
    def _filter_by_year(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Filter dataframe by year."""
        year_mask = (
            (df['CompletionYear'] == year) |
            (df['FundingYear'] == year) |
            (df['InfraYear'] == year)
        )
        return df[year_mask]
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query to determine processing approach."""
        # Metadata/system queries (highest priority)
        if any(term in query for term in ['how many total', 'total projects', 'dataset', 'unique values', 'missing data', 'columns', 'fields']):
            return 'metadata_analysis'
        
        # Analysis/insights queries (high priority)  
        if any(term in query for term in ['distribution', 'trend', 'growth', 'statistics', 'insights']):
            return 'analysis_insights'
        
        # Contractor-related queries (check first for priority)
        contractor_terms = [
            'contractor', 'company', 'builder', 'construction', 'who built', 
            'who constructed', 'top contractor', 'most projects', 'which contractor',
            'built by', 'constructed by', 'developer', 'firm'
        ]
        if any(term in query for term in contractor_terms):
            return 'contractor_analysis'
        
        # Cost-related queries (expanded patterns)
        cost_terms = [
            'expensive', 'cost', 'budget', 'price', 'amount', 'million', 'billion', 
            'cheapest', 'affordable', 'average', 'spending', 'investment',
            'least', 'lowest', 'smallest', 'minimum', 'maximum', 'highest', 'most',
            'low budget', 'high budget', 'low cost', 'high cost'
        ]
        if any(term in query for term in cost_terms):
            return 'cost_analysis'
        
        # Time/completion-related queries (expanded patterns)
        time_terms = [
            'completed', 'finished', 'done', 'completion', 'when', 'recent', 
            'latest', 'oldest', 'first', 'last', 'year', 'date', 'timeline',
            'started', 'began', 'ongoing', 'in progress'
        ]
        if any(term in query for term in time_terms):
            return 'completion_analysis'
        
        # Project type queries (expanded patterns)
        type_terms = [
            'type', 'kind', 'drainage', 'bridge', 'seawall', 'revetment', 
            'mitigation', 'flood control', 'pumping', 'embankment', 'dike',
            'canal', 'culvert', 'slope protection', 'river control'
        ]
        if any(term in query for term in type_terms):
            return 'project_type_analysis'
        
        # Location-based queries (expanded patterns)
        location_terms = [
            'where', 'location', 'region', 'province', 'city', 'municipality', 
            'how many', 'count', 'number of', 'investment', 'total cost',
            'in', 'at', 'near', 'around', 'barangay', 'area'
        ]
        if any(term in query for term in location_terms):
            return 'location_analysis'
        
        # Comparison queries (expanded patterns)
        comparison_terms = [
            'compare', 'vs', 'versus', 'difference', 'between', 'comparison',
            'better', 'worse', 'more than', 'less than', 'against'
        ]
        if any(term in query for term in comparison_terms):
            return 'comparison'
        
        
        # Default to general search
        else:
            return 'general'
    
    def _handle_cost_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle cost-related queries."""
        if df.empty:
            return []
        
        # Check if ContractCost column exists
        if 'ContractCost' not in df.columns:
            return []
        
        # Filter out records with no cost data
        cost_df = df[df['ContractCost'].notna() & (df['ContractCost'] > 0)]
        
        if cost_df.empty:
            # Fallback to all records if no cost data
            cost_df = df
        
        # Check if this is an average/summary query
        if any(term in query for term in ['average', 'total', 'summary', 'spending']):
            # Return all records for summary analysis, but mark as summary type
            result_records = self._convert_to_records(cost_df.head(top_k), 1.0)
            
            # Add summary statistics to the first record
            if result_records:
                total_cost = cost_df['ContractCost'].sum()
                avg_cost = cost_df['ContractCost'].mean()
                project_count = len(cost_df)
                
                # Add summary info to all records
                for record in result_records:
                    record['total_investment'] = total_cost
                    record['average_cost'] = avg_cost
                    record['project_count'] = project_count
                    record['query_type'] = 'cost_summary'
            
            return result_records
        
        elif any(term in query for term in ['cheapest', 'lowest', 'least', 'smallest', 'minimum', 'low budget', 'low cost']):
            # Sort by contract cost ascending (cheapest first)
            sorted_df = cost_df.sort_values('ContractCost', ascending=True)
        elif any(term in query for term in ['expensive', 'highest', 'most', 'largest', 'maximum', 'high budget', 'high cost']):
            # Sort by contract cost descending (most expensive first)
            sorted_df = cost_df.sort_values('ContractCost', ascending=False)
        else:
            # Default to expensive for general cost queries
            sorted_df = cost_df.sort_values('ContractCost', ascending=False)
        
        return self._convert_to_records(sorted_df.head(top_k), 1.0)
    
    def _handle_contractor_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle contractor-related queries."""
        if 'most projects' in query or 'top contractor' in query:
            # Group by contractor and count projects
            contractor_counts = df['Contractor'].value_counts()
            top_contractors = contractor_counts.head(top_k).index.tolist()
            
            # Get sample projects from top contractors
            results = []
            for contractor in top_contractors:
                contractor_projects = df[df['Contractor'] == contractor]
                # Get the most expensive project from this contractor
                top_project = contractor_projects.sort_values('ContractCost', ascending=False, ).iloc[0]
                record = top_project.to_dict()
                record['similarity_score'] = 1.0
                record['contractor_project_count'] = contractor_counts[contractor]
                results.append(record)
            return results
        else:
            # Search for specific contractor
            return self._semantic_search(df, query, top_k)
    
    def _handle_completion_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle completion-related queries with advanced filtering."""
        if df.empty:
            return []
        
        # Filter out records with no completion data
        completion_df = df[df['CompletionYear'].notna() | df['CompletionDateActual'].notna()]
        
        if completion_df.empty:
            completion_df = df  # Fallback to all records
        
        # Enhanced pattern matching for completion queries
        recent_terms = ['recent', 'latest', 'new', 'current', '2024', '2023', '2022']
        old_terms = ['oldest', 'first', 'early', 'initial', '2021', '2020', '2019']
        
        if any(term in query for term in recent_terms):
            # Sort by completion date descending (most recent first)
            if 'CompletionDateActual' in completion_df.columns:
                sorted_df = completion_df.sort_values(['CompletionYear', 'CompletionDateActual'], 
                                                    ascending=[False, False])
            else:
                sorted_df = completion_df.sort_values('CompletionYear', ascending=False)
        elif any(term in query for term in old_terms):
            # Sort by completion date ascending (oldest first)
            if 'CompletionDateActual' in completion_df.columns:
                sorted_df = completion_df.sort_values(['CompletionYear', 'CompletionDateActual'], 
                                                    ascending=[True, True])
            else:
                sorted_df = completion_df.sort_values('CompletionYear', ascending=True)
        else:
            # Default to recent, but also consider cost for relevance
            sorted_df = completion_df.sort_values(['CompletionYear', 'ContractCost'], 
                                                ascending=[False, False])
        
        return self._convert_to_records(sorted_df.head(top_k), 1.0)
    
    def _handle_project_type_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle project type queries with comprehensive filtering."""
        if df.empty:
            return []
        
        type_filtered = df.copy()
        
        # Enhanced project type matching
        project_type_patterns = {
            'drainage': ['drainage', 'storm water', 'culvert', 'canal'],
            'bridge': ['bridge', 'viaduct', 'overpass', 'underpass'],
            'seawall': ['seawall', 'sea wall', 'coastal protection', 'breakwater'],
            'revetment': ['revetment', 'riprapping', 'riprap', 'slope protection'],
            'flood control': ['flood control', 'flood mitigation', 'flood management'],
            'river': ['river', 'waterway', 'creek', 'stream'],
            'embankment': ['embankment', 'dike', 'levee'],
            'pumping': ['pumping station', 'pump house', 'drainage pump']
        }
        
        # Find matching project types
        matched_types = []
        for type_name, patterns in project_type_patterns.items():
            if any(pattern in query.lower() for pattern in patterns):
                matched_types.append(type_name)
        
        # Apply filters based on matched types
        if matched_types:
            type_masks = []
            for type_name in matched_types:
                patterns = project_type_patterns[type_name]
                type_mask = pd.Series([False] * len(df), index=df.index)
                
                for pattern in patterns:
                    pattern_mask = (
                        df['TypeofWork'].str.contains(pattern, case=False, na=False) |
                        df['ProjectDescription'].str.contains(pattern, case=False, na=False) |
                        df['infra_type'].str.contains(pattern, case=False, na=False)
                    )
                    type_mask = type_mask | pattern_mask
                
                type_masks.append(type_mask)
            
            # Combine all type masks
            combined_mask = pd.Series([False] * len(df), index=df.index)
            for mask in type_masks:
                combined_mask = combined_mask | mask
            
            type_filtered = df[combined_mask]
        
        # If no specific type filtering worked, try semantic search
        if type_filtered.empty or len(type_filtered) == len(df):
            return self._semantic_search(df, query, top_k)
        
        # Sort by relevance (cost + completion year)
        if 'ContractCost' in type_filtered.columns and 'CompletionYear' in type_filtered.columns:
            # Multi-criteria sorting: recent projects with higher costs first
            sorted_df = type_filtered.sort_values(['CompletionYear', 'ContractCost'], 
                                                ascending=[False, False], )
        elif 'ContractCost' in type_filtered.columns:
            sorted_df = type_filtered.sort_values('ContractCost', ascending=False, )
        else:
            sorted_df = type_filtered
        
        return self._convert_to_records(sorted_df.head(top_k), 1.0)
    
    def _handle_location_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle location-based queries with comprehensive analysis."""
        if df.empty:
            return []
        
        # Determine what aspect of location is being asked
        if any(term in query.lower() for term in ['how many', 'count', 'number of']):
            # Count-based queries - group by location
            return self._handle_location_count_queries(df, query, top_k)
        elif any(term in query.lower() for term in ['total cost', 'investment', 'budget', 'spending']):
            # Investment analysis by location
            return self._handle_location_investment_queries(df, query, top_k)
        else:
            # Default: show top projects by cost within location
            sorted_df = df.sort_values(['ContractCost', 'CompletionYear'], 
                                     ascending=[False, False], )
            return self._convert_to_records(sorted_df.head(top_k), 1.0)
    
    def _handle_location_count_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle location count queries."""
        # Group by municipality and count projects
        location_counts = df.groupby(['Municipality', 'Province']).agg({
            'ProjectDescription': 'count',
            'ContractCost': ['sum', 'mean'],
            'CompletionYear': ['min', 'max']
        }).round(2)
        
        location_counts.columns = ['project_count', 'total_cost', 'avg_cost', 'first_year', 'last_year']
        location_counts = location_counts.reset_index()
        location_counts = location_counts.sort_values('project_count', ascending=False)
        
        # Convert to records format
        results = []
        for _, row in location_counts.head(top_k).iterrows():
            record = {
                'Municipality': row['Municipality'],
                'Province': row['Province'],
                'project_count': int(row['project_count']),
                'total_investment': float(row['total_cost']) if pd.notna(row['total_cost']) else 0,
                'average_cost': float(row['avg_cost']) if pd.notna(row['avg_cost']) else 0,
                'first_project_year': int(row['first_year']) if pd.notna(row['first_year']) else None,
                'last_project_year': int(row['last_year']) if pd.notna(row['last_year']) else None,
                'similarity_score': 1.0
            }
            results.append(record)
        
        return results
    
    def _handle_location_investment_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle location investment analysis queries."""
        # Group by location and sum investments
        investment_analysis = df.groupby(['Municipality', 'Province', 'Region']).agg({
            'ContractCost': ['sum', 'count', 'mean', 'max'],
            'CompletionYear': ['min', 'max']
        }).round(2)
        
        investment_analysis.columns = ['total_investment', 'project_count', 'avg_cost', 'max_cost', 'first_year', 'last_year']
        investment_analysis = investment_analysis.reset_index()
        investment_analysis = investment_analysis.sort_values('total_investment', ascending=False)
        
        # Convert to records format
        results = []
        for _, row in investment_analysis.head(top_k).iterrows():
            record = {
                'Municipality': row['Municipality'],
                'Province': row['Province'],
                'Region': row['Region'],
                'total_investment': float(row['total_investment']) if pd.notna(row['total_investment']) else 0,
                'project_count': int(row['project_count']),
                'average_project_cost': float(row['avg_cost']) if pd.notna(row['avg_cost']) else 0,
                'largest_project_cost': float(row['max_cost']) if pd.notna(row['max_cost']) else 0,
                'investment_period': f"{int(row['first_year']) if pd.notna(row['first_year']) else 'N/A'}-{int(row['last_year']) if pd.notna(row['last_year']) else 'N/A'}",
                'similarity_score': 1.0
            }
            results.append(record)
        
        return results
    
    def _handle_comparison_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle comparison queries with advanced analysis."""
        if df.empty:
            return []
        
        # Extract comparison entities from query
        comparison_terms = self._extract_comparison_terms(query.lower())
        
        if len(comparison_terms) >= 2:
            # Compare specific entities
            return self._handle_specific_comparisons(df, comparison_terms, top_k)
        else:
            # General comparison - show diverse top projects
            return self._handle_general_comparisons(df, query, top_k)
    
    def _extract_comparison_terms(self, query: str) -> list:
        """Extract entities being compared from query."""
        # Common comparison patterns
        comparison_patterns = [
            r'between\s+(.+?)\s+and\s+(.+?)(?:\s|$)',
            r'(.+?)\s+vs\s+(.+?)(?:\s|$)',
            r'(.+?)\s+versus\s+(.+?)(?:\s|$)',
            r'compare\s+(.+?)\s+(?:and|with)\s+(.+?)(?:\s|$)'
        ]
        
        import re
        for pattern in comparison_patterns:
            match = re.search(pattern, query)
            if match:
                return [term.strip() for term in match.groups()]
        
        return []
    
    def _handle_specific_comparisons(self, df: pd.DataFrame, terms: list, top_k: int) -> List[Dict[str, Any]]:
        """Handle specific entity comparisons."""
        results = []
        
        for term in terms[:2]:  # Compare first two terms
            # Search for projects matching this term
            term_mask = (
                df['Municipality'].str.contains(term, case=False, na=False) |
                df['Province'].str.contains(term, case=False, na=False) |
                df['Region'].str.contains(term, case=False, na=False) |
                df['Contractor'].str.contains(term, case=False, na=False) |
                df['TypeofWork'].str.contains(term, case=False, na=False)
            )
            
            term_projects = df[term_mask]
            if not term_projects.empty:
                # Get top project for this term
                top_project = term_projects.sort_values('ContractCost', ascending=False, ).iloc[0]
                record = top_project.to_dict()
                record['comparison_term'] = term
                record['similarity_score'] = 1.0
                results.append(record)
        
        return results[:top_k]
    
    def _handle_general_comparisons(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle general comparison queries."""
        # Return diverse projects for comparison
        if 'region' in query.lower():
            # Compare by regions
            regions = df['Region'].value_counts().head(top_k).index
            results = []
            for region in regions:
                region_projects = df[df['Region'] == region]
                if not region_projects.empty:
                    top_project = region_projects.sort_values('ContractCost', ascending=False, ).iloc[0]
                    record = top_project.to_dict()
                    record['comparison_category'] = 'region'
                    record['similarity_score'] = 1.0
                    results.append(record)
            return results
        else:
            # General top projects for comparison
            sorted_df = df.sort_values('ContractCost', ascending=False, )
            return self._convert_to_records(sorted_df.head(top_k), 1.0)
    
    def _semantic_search(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic search using TF-IDF."""
        if df.empty:
            return []
        
        # Create search texts for filtered dataframe
        search_texts = []
        for _, row in df.iterrows():
            text_parts = []
            for col in self.text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            search_texts.append(" ".join(text_parts))
        
        if not search_texts:
            return []
        
        # Create temporary vectorizer for filtered data
        temp_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        temp_tfidf_matrix = temp_vectorizer.fit_transform(search_texts)
        
        # Transform query and calculate similarities
        query_vector = temp_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, temp_tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for better results
                record = df.iloc[idx].to_dict()
                record['similarity_score'] = similarities[idx]
                results.append(record)
        
        return results
    
    def _convert_to_records(self, df: pd.DataFrame, score: float) -> List[Dict[str, Any]]:
        """Convert dataframe to list of records with similarity score."""
        records = []
        for _, row in df.iterrows():
            record = row.to_dict()
            record['similarity_score'] = score
            records.append(record)
        return records
    
    def _handle_metadata_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle metadata and system information queries."""
        if df.empty:
            return []
        
        # Create a metadata summary record
        metadata_info = {
            'query_type': 'metadata',
            'total_projects': len(df),
            'total_columns': len(df.columns),
            'columns_list': list(df.columns),
            'unique_regions': df['Region'].nunique() if 'Region' in df.columns else 0,
            'unique_provinces': df['Province'].nunique() if 'Province' in df.columns else 0,
            'unique_municipalities': df['Municipality'].nunique() if 'Municipality' in df.columns else 0,
            'unique_contractors': df['Contractor'].nunique() if 'Contractor' in df.columns else 0,
            'date_range_start': df['InfraYear'].min() if 'InfraYear' in df.columns else None,
            'date_range_end': df['InfraYear'].max() if 'InfraYear' in df.columns else None,
            'missing_data_summary': df.isnull().sum().to_dict(),
            'similarity_score': 1.0
        }
        
        # Add specific query responses
        if 'unique values' in query:
            if 'infra_type' in query:
                metadata_info['unique_infra_types'] = df['infra_type'].unique().tolist() if 'infra_type' in df.columns else []
            elif 'contractor' in query:
                metadata_info['unique_contractors_list'] = df['Contractor'].unique().tolist() if 'Contractor' in df.columns else []
        
        return [metadata_info]
    
    def _handle_analysis_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle analysis and insights queries."""
        if df.empty:
            return []
        
        # Create analysis summary
        analysis_info = {
            'query_type': 'analysis',
            'similarity_score': 1.0
        }
        
        # Distribution analysis
        if 'distribution' in query:
            if 'region' in query:
                analysis_info['region_distribution'] = df['Region'].value_counts().to_dict()
            elif 'province' in query:
                analysis_info['province_distribution'] = df['Province'].value_counts().head(10).to_dict()
            elif 'year' in query:
                analysis_info['year_distribution'] = df['CompletionYear'].value_counts().sort_index().to_dict()
        
        # Trend analysis
        if 'trend' in query and 'funding' in query:
            if 'ContractCost' in df.columns and 'InfraYear' in df.columns:
                yearly_funding = df.groupby('InfraYear')['ContractCost'].sum().to_dict()
                analysis_info['funding_trend'] = yearly_funding
        
        # Top analysis
        if 'top' in query:
            if 'expensive' in query:
                top_projects = df.nlargest(5, 'ContractCost')[['ProjectDescription', 'ContractCost', 'Municipality', 'Province']].to_dict('records')
                analysis_info['top_expensive_projects'] = top_projects
            elif 'contractor' in query:
                top_contractors = df['Contractor'].value_counts().head(5).to_dict()
                analysis_info['top_contractors'] = top_contractors
        
        return [analysis_info]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dataset.
        
        Returns:
            Dict containing various statistics about the dataset including
            record counts, column information, and data quality metrics.
        """
        if self.df is None or self.df.empty:
            return {
                'total_records': 0,
                'columns': [],
                'numeric_columns': [],
                'text_columns': [],
                'missing_data': {},
                'unique_regions': 0,
                'unique_provinces': 0,
                'unique_contractors': 0,
                'date_range': 'N/A'
            }
        
        return {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': self.text_columns,
            'missing_data': self.df.isnull().sum().to_dict(),
            'unique_regions': self.df['Region'].nunique() if 'Region' in self.df.columns else 0,
            'unique_provinces': self.df['Province'].nunique() if 'Province' in self.df.columns else 0,
            'unique_contractors': self.df['Contractor'].nunique() if 'Contractor' in self.df.columns else 0,
            'date_range': f"{self.df['InfraYear'].min()}-{self.df['InfraYear'].max()}" if 'InfraYear' in self.df.columns else 'N/A'
        }
    
    def get_column_info(self) -> Dict[str, List[str]]:
        """Get information about available columns and sample values."""
        if self.df is None:
            return {}
        
        column_info = {}
        return {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': self.text_columns,
            'missing_data': self.df.isnull().sum().to_dict()
        }
    
    def filter_records(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataset."""
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        for column, value in filters.items():
            if column in filtered_df.columns and value:
                if filtered_df[column].dtype == 'object':
                    # Text-based filtering (case-insensitive partial match)
                    filtered_df = filtered_df[
                        filtered_df[column].str.contains(str(value), case=False, na=False)
                    ]
                else:
                    # Numeric filtering
                    try:
                        filtered_df = filtered_df[filtered_df[column] == float(value)]
                    except (ValueError, TypeError):
                        continue
        
        return filtered_df
