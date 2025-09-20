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
            
            # st.success(f"Successfully loaded {len(self.df):,} flood control project records from {source_description}.")  # Commented out to hide message
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
            
            # Extract year mentions
            year = self._extract_year(query_lower)
            if year:
                filtered_df = self._filter_by_year(filtered_df, year)
            
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
            else:
                # Default semantic search
                return self._semantic_search(filtered_df, query, top_k)
            
        except Exception as e:
            st.error(f"Error searching records: {str(e)}")
            return []
    
    def _extract_location(self, query: str) -> str:
        """Extract location mentions from query."""
        # Common Philippine locations
        locations = {
            'cebu': ['cebu', 'cebu city'],
            'manila': ['manila', 'metro manila', 'ncr'],
            'davao': ['davao', 'davao city'],
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
            'region xii': ['region xii', 'region 12', 'soccsksargen'],
            'region xiii': ['region xiii', 'region 13', 'caraga'],
            'barmm': ['barmm', 'bangsamoro', 'armm'],
            'car': ['car', 'cordillera']
        }
        
        for standard_name, variants in locations.items():
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
        """Filter dataframe by location."""
        location_mask = (
            df['Municipality'].str.contains(location, case=False, na=False) |
            df['Province'].str.contains(location, case=False, na=False) |
            df['Region'].str.contains(location, case=False, na=False)
        )
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
        if any(term in query for term in ['expensive', 'cost', 'budget', 'price', 'amount', 'million', 'billion']):
            return 'cost_analysis'
        elif any(term in query for term in ['contractor', 'company', 'builder', 'construction', 'who built', 'who constructed']):
            return 'contractor_analysis'
        elif any(term in query for term in ['completed', 'finished', 'done', 'completion', 'when', 'year']):
            return 'completion_analysis'
        elif any(term in query for term in ['type', 'kind', 'drainage', 'bridge', 'seawall', 'revetment', 'mitigation']):
            return 'project_type_analysis'
        elif any(term in query for term in ['where', 'location', 'region', 'province', 'city', 'municipality']):
            return 'location_analysis'
        elif any(term in query for term in ['compare', 'vs', 'versus', 'difference', 'between']):
            return 'comparison'
        else:
            return 'general'
    
    def _handle_cost_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle cost-related queries."""
        if 'expensive' in query or 'highest' in query:
            # Sort by contract cost descending
            sorted_df = df.sort_values('ContractCost', ascending=False, na_last=True)
        elif 'cheapest' in query or 'lowest' in query:
            # Sort by contract cost ascending
            sorted_df = df.sort_values('ContractCost', ascending=True, na_last=True)
        else:
            # Default to expensive
            sorted_df = df.sort_values('ContractCost', ascending=False, na_last=True)
        
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
                top_project = contractor_projects.sort_values('ContractCost', ascending=False, na_last=True).iloc[0]
                record = top_project.to_dict()
                record['similarity_score'] = 1.0
                record['contractor_project_count'] = contractor_counts[contractor]
                results.append(record)
            return results
        else:
            # Search for specific contractor
            return self._semantic_search(df, query, top_k)
    
    def _handle_completion_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle completion-related queries."""
        if 'recent' in query or 'latest' in query:
            # Sort by completion date descending
            sorted_df = df.sort_values('CompletionDateActual', ascending=False, na_last=True)
        elif 'oldest' in query or 'first' in query:
            # Sort by completion date ascending
            sorted_df = df.sort_values('CompletionDateActual', ascending=True, na_last=True)
        else:
            # Default to recent
            sorted_df = df.sort_values('CompletionYear', ascending=False, na_last=True)
        
        return self._convert_to_records(sorted_df.head(top_k), 1.0)
    
    def _handle_project_type_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle project type queries."""
        # Filter by project type
        type_filtered = df.copy()
        
        if 'drainage' in query:
            type_filtered = df[df['TypeofWork'].str.contains('drainage', case=False, na=False)]
        elif 'bridge' in query:
            type_filtered = df[df['TypeofWork'].str.contains('bridge', case=False, na=False)]
        elif 'seawall' in query:
            type_filtered = df[df['TypeofWork'].str.contains('seawall', case=False, na=False)]
        elif 'revetment' in query or 'riprapping' in query:
            type_filtered = df[df['TypeofWork'].str.contains('revetment|riprapping', case=False, na=False)]
        
        # Sort by cost if we have type-filtered results
        if not type_filtered.empty:
            sorted_df = type_filtered.sort_values('ContractCost', ascending=False, na_last=True)
            return self._convert_to_records(sorted_df.head(top_k), 1.0)
        else:
            return self._semantic_search(df, query, top_k)
    
    def _handle_location_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle location-based queries."""
        # Sort by cost within the location
        sorted_df = df.sort_values('ContractCost', ascending=False, na_last=True)
        return self._convert_to_records(sorted_df.head(top_k), 1.0)
    
    def _handle_comparison_queries(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Handle comparison queries."""
        # For now, return top projects by cost for comparison
        sorted_df = df.sort_values('ContractCost', ascending=False, na_last=True)
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
    
    def get_column_info(self) -> Dict[str, List[str]]:
        """Get information about available columns and sample values."""
        if self.df is None:
            return {}
        
        column_info = {}
        for col in self.df.columns:
            # Get unique values (limited to first 10 for display)
            unique_vals = self.df[col].dropna().unique()[:10]
            column_info[col] = [str(val) for val in unique_vals]
        
        return column_info
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        if self.df is None:
            return {}
        
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
