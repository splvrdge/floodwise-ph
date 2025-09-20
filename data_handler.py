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
            
            st.success(f"Successfully loaded {len(self.df):,} flood control project records from {source_description}.")
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
        """Search for records most relevant to the query."""
        if self.df is None or self.vectorizer is None:
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k most similar records
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_records = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    record = self.df.iloc[idx].to_dict()
                    record['similarity_score'] = similarities[idx]
                    relevant_records.append(record)
            
            return relevant_records
            
        except Exception as e:
            st.error(f"Error searching records: {str(e)}")
            return []
    
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
