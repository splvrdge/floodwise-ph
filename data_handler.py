import logging
import re
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# For fuzzy string matching
try:
    from fuzzywuzzy import fuzz
except ImportError:
    logger.warning("fuzzywuzzy not found. Using simple string matching instead.")
    # Fallback to a simple string matching if fuzzywuzzy is not available
    class SimpleFuzz:
        @staticmethod
        def token_sort_ratio(a, b):
            return 100 if a.lower() == b.lower() else 0
    
    fuzz = SimpleFuzz()

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
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the loaded dataset.
        
        Returns:
            dict: A dictionary containing various statistics about the dataset
        """
        if self.df is None or self.df.empty:
            return {}
            
        stats = {
            'total_records': len(self.df),
            'columns': self.df.columns.tolist(),
            'column_count': len(self.df.columns),
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicate_rows': int(self.df.duplicated().sum())
        }
        
        # Add date range if there are date columns
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        if not date_columns.empty:
            for col in date_columns:
                stats[f'min_date_{col}'] = str(self.df[col].min())
                stats[f'max_date_{col}'] = str(self.df[col].max())
                
        # Add numeric stats
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        stats['numeric_columns'] = numeric_columns.tolist()
        
        # Add categorical stats
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        stats['categorical_columns'] = categorical_columns.tolist()
        
        # Get unique values for categorical columns (limited to first 10 for performance)
        for col in categorical_columns[:10]:  # Limit to first 10 categorical columns
            unique_vals = self.df[col].dropna().unique()
            stats[f'unique_{col}'] = len(unique_vals)
            if len(unique_vals) <= 20:  # Only include values if not too many
                stats[f'unique_{col}_values'] = unique_vals.tolist()
                
        return stats
    
    def _standardize_text(self, text: str) -> str:
        """Standardize text by converting to lowercase and removing extra whitespace."""
        if pd.isna(text):
            return ''
        return ' '.join(str(text).strip().lower().split())

    def _extract_project_type(self, description: str) -> str:
        """Extract and standardize project type from description."""
        if pd.isna(description):
            return 'Other'
            
        description = description.lower()
        
        # Define project type patterns
        type_patterns = {
            'flood_control': r'flood control|flood mitigation|river control|riverbank protection|dike|levee|embankment',
            'drainage': r'drainage|culvert|canal|waterway|catch basin',
            'bridge': r'bridge|footbridge|viaduct',
            'road': r'road|highway|pavement|asphalt|concrete',
            'retention': r'retention basin|detention pond|catchment',
            'seawall': r'seawall|shoreline|coastal protection',
            'slope': r'slope protection|erosion control|landslide',
            'pump': r'pump|pumping station',
            'floodgate': r'floodgate|tide gate',
            'revetment': r'revetment|riprap|gabion'
        }
        
        # Check for each pattern
        for type_name, pattern in type_patterns.items():
            if re.search(pattern, description):
                return type_name
                
        return 'other'

    def _parse_cost(self, cost_str: str) -> float:
        """Parse cost string into numeric value."""
        if pd.isna(cost_str):
            return 0.0
            
        # Handle numeric values
        if isinstance(cost_str, (int, float)):
            return float(cost_str)
            
        # Clean the string
        cost_str = str(cost_str).lower().replace('₱', '').replace(',', '').strip()
        
        # Handle currency abbreviations
        multiplier = 1.0
        if 'm' in cost_str or 'million' in cost_str:
            multiplier = 1000000
            cost_str = cost_str.replace('m', '').replace('million', '')
        elif 'b' in cost_str or 'billion' in cost_str:
            multiplier = 1000000000
            cost_str = cost_str.replace('b', '').replace('billion', '')
        elif 'k' in cost_str or 'thousand' in cost_str:
            multiplier = 1000
            cost_str = cost_str.replace('k', '').replace('thousand', '')
            
        # Extract numeric value
        try:
            return float(re.sub(r'[^0-9.]', '', cost_str)) * multiplier
        except (ValueError, TypeError):
            return 0.0

    def _parse_date(self, date_str, date_format=None):
        """Parse date string into datetime object with flexible format handling."""
        if pd.isna(date_str):
            return None
            
        if isinstance(date_str, (int, float)) and not pd.isna(date_str):
            # Handle Unix timestamp (in milliseconds)
            try:
                return datetime.fromtimestamp(date_str / 1000)
            except (ValueError, TypeError):
                return None
                
        # Try parsing with common date formats
        date_formats = [
            '%m/%d/%Y', '%Y-%m-%d', '%d-%b-%y', '%b %d, %Y', 
            '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y'
        ]
        
        if date_format:
            date_formats.insert(0, date_format)
            
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except (ValueError, TypeError):
                continue
                
        return None

    def _process_loaded_data(self, source_description: str) -> bool:
        """Process loaded data regardless of source."""
        try:
            # Basic validation
            if self.df.empty:
                st.error(f"The {source_description} is empty.")
                return False
            
            # Standardize text columns
            text_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            for col in text_columns:
                self.df[col] = self.df[col].apply(self._standardize_text)
            
            # Add standardized location columns
            self.df['location_standard'] = self.df['Municipality'].fillna('') + ', ' + \
                                         self.df['Province'].fillna('') + ', ' + \
                                         self.df['Region'].fillna('')
            
            # Extract and standardize project types
            self.df['project_type'] = self.df['ProjectDescription'].apply(self._extract_project_type)
            
            # Parse costs
            cost_columns = [col for col in self.df.columns if 'cost' in col.lower() or 'abc' in col.lower()]
            for col in cost_columns:
                self.df[f'{col}_numeric'] = self.df[col].apply(self._parse_cost)
            
            # Parse dates
            date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'year' in col.lower()]
            for col in date_columns:
                if 'year' in col.lower() and self.df[col].dtype == 'object':
                    # Extract year from string if needed
                    self.df[col] = self.df[col].str.extract(r'(\d{4})', expand=False)
                self.df[f'{col}_parsed'] = self.df[col].apply(self._parse_date)
            
            # Create searchable text by combining all text columns
            self.text_columns = text_columns
            self._prepare_search_index()
            
            # Cache unique values for filtering
            self._cache_unique_values()
            
            return True
            
        except Exception as e:
            st.error(f"Error processing data from {source_description}: {str(e)}")
            st.error(traceback.format_exc())
            return False
            
    def _cache_unique_values(self):
        """Cache unique values for filtering."""
        self.unique_values_cache = {}
        for col in self.df.columns:
            if self.df[col].nunique() < 1000:  # Only cache for columns with reasonable number of unique values
                self.unique_values_cache[col] = self.df[col].dropna().unique().tolist()
    
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
    
    def search_relevant_records(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for records most relevant to the query with intelligent processing.
        
        This method implements a multi-stage search strategy:
        1. Parse the query for location, year, and other filters
        2. Apply filters to narrow down the dataset
        3. Use TF-IDF for semantic search if needed
        4. Fall back to simple text matching if necessary
        
        Args:
            query: The search query
            top_k: Maximum number of results to return (default: 10)
            
        Returns:
            List of matching records as dictionaries
            
        Raises:
            ValueError: If no data is available
        """
        try:
            # Input validation
            if not hasattr(self, 'df') or self.df is None or self.df.empty:
                error_msg = "No data available. Please load the dataset first."
                logger.error(error_msg)
                if st.session_state.get('debug_mode', False):
                    st.error(error_msg)
                return []
                
            if not query or not isinstance(query, str) or not query.strip():
                logger.warning("Empty or invalid query provided")
                return []
                
            logger.info(f"Processing search query: {query}")
            
            # Initialize working dataframe
            working_df = self.df.copy()
            
            # Extract location and year information
            try:
                location_terms = self._extract_location_terms(query)
                year, date_range = self._extract_years(query)
                
                logger.debug(f"Extracted location terms: {location_terms}")
                logger.debug(f"Extracted year: {year}, date range: {date_range}")
                
                # Apply location filter if terms were found
                if location_terms:
                    working_df = self._filter_by_location(working_df, location_terms)
                    logger.debug(f"After location filter: {len(working_df)} records")
                    
                # Apply year/date filter if specified
                if year is not None or (date_range and any(date_range)):
                    working_df = self._filter_by_year(working_df, year, date_range)
                    logger.debug(f"After year filter: {len(working_df)} records")
                    
            except Exception as e:
                error_msg = f"Error applying filters: {str(e)}"
                logger.error(error_msg, exc_info=True)
                if st.session_state.get('debug_mode', False):
                    st.warning(f"Warning: {error_msg}")
                # Continue with unfiltered results if there's an error in filtering
                working_df = self.df.copy()
            
            # If no results after filtering, try a broader search
            if len(working_df) == 0:
                logger.info("No matching records found after filtering, trying broader search")
                working_df = self.df.copy()
                
            # If we still have too many results, use TF-IDF to find the most relevant ones
            if len(working_df) > top_k:
                try:
                    # Combine text columns for TF-IDF
                    text_columns = [col for col in ['ProjectName', 'ProjectDescription', 'Location', 'Contractor'] 
                                  if col in working_df.columns]
                    
                    if text_columns:
                        working_df['combined_text'] = working_df[text_columns].fillna('').apply(
                            lambda x: ' '.join(x.astype(str)), axis=1
                        )
                        
                        # Create TF-IDF vectors
                        vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(working_df['combined_text'])
                        query_vec = vectorizer.transform([query])
                        
                        # Calculate cosine similarity
                        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                        
                        # Get top-k most similar documents
                        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
                        working_df = working_df.iloc[top_indices]
                        
                        logger.debug(f"Applied TF-IDF ranking, top {len(working_df)} results")
                    
                except Exception as e:
                    error_msg = f"Error in TF-IDF processing: {str(e)}"
                    logger.warning(error_msg, exc_info=True)
                    if st.session_state.get('debug_mode', False):
                        st.warning(f"Debug: {error_msg}")
                    # Fall back to simple text matching
                    working_df = working_df.head(top_k)
            
            # Convert results to list of dictionaries
            results = working_df.head(top_k).to_dict('records')
            logger.info(f"Returning {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = f"Unexpected error in search_relevant_records: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if st.session_state.get('debug_mode', False):
                st.error(f"Error: {error_msg}")
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
        years = re.findall(r'\b(20[0-9]{2})\b', query)
        return int(years[0]) if years else None
    
    def _normalize_location_name(self, name: str) -> str:
        """Normalize location names for consistent matching."""
        if pd.isna(name):
            return ''
            
        # Common replacements and normalizations
        replacements = {
            'city of ': '',
            'city': '',
            'municipality of ': '',
            'mun. of ': '',
            'province of ': '',
            'prov. of ': '',
            'north': 'n',
            'south': 's',
            'east': 'e',
            'west': 'w',
            'northern': 'n',
            'southern': 's',
            'eastern': 'e',
            'western': 'w',
            'st.': 'saint',
            'mt.': 'mount',
            'ft.': 'fort'
        }
        
        name = str(name).strip().lower()
        
        # Apply replacements
        for old, new in replacements.items():
            name = name.replace(old.lower(), new.lower())
            
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Standardize region names
        region_mapping = {
            'ncr': 'national capital region',
            'car': 'cordillera administrative region',
            'armm': 'autonomous region in muslim mindanao',
            'barmm': 'bangsamoro autonomous region in muslim mindanao'
        }
        
        for abbr, full in region_mapping.items():
            if name == abbr.lower() or name == full.lower():
                return full.lower()
                
        # Standardize province names
        province_mapping = {
            'cotabato (north cotabato)': 'north cotabato',
            'cotabato': 'north cotabato',
            'davao del norte': 'davao del norte',
            'davao del sur': 'davao del sur',
            'davao oriental': 'davao oriental',
            'davao de oro': 'davao de oro',
            'davao occidental': 'davao occidental'
        }
        
        return province_mapping.get(name, name)

    def _is_location_match(self, text: str, search_terms: list) -> bool:
        """Check if text matches any of the search terms with fuzzy matching."""
        if pd.isna(text):
            return False
            
        text = str(text).lower()
        
        for term in search_terms:
            # Exact match
            if term.lower() == text:
                return True
                
            # Substring match
            if term.lower() in text or text in term.lower():
                return True
                
            # Fuzzy matching for slight misspellings
            if fuzz.token_sort_ratio(term.lower(), text) > 85:  # 85% similarity threshold
                return True
                
            # Check for common abbreviations
            if len(term) > 3 and len(text) > 3:  # Only for terms longer than 3 chars
                if fuzz.ratio(term[:4].lower(), text[:4].lower()) > 90:  # Check first 4 chars
                    return True
                    
        return False

    def _filter_by_location(self, df: pd.DataFrame, location: str) -> pd.DataFrame:
        """
        Filter dataframe by location with flexible and precise matching.
        
        Args:
            df: DataFrame to filter
            location: Location name to filter by (can be city, municipality, province, or region)
            
        Returns:
            Filtered DataFrame containing only matching locations
        """
        if df.empty or not location or not isinstance(location, str):
            return df
            
        # Normalize the search location
        location = self._normalize_location_name(location)
        
        # If no location after normalization, return empty DataFrame
        if not location:
            return pd.DataFrame()
            
        # Create a copy to avoid SettingWithCopyWarning
        df_filtered = df.copy()
        
        # Get location hierarchy (region > province > city/municipality > barangay)
        location_columns = ['Region', 'Province', 'Municipality', 'Barangay', 'City']
        available_columns = [col for col in location_columns if col in df_filtered.columns]
        
        # Create search terms with variations
        search_terms = [location]
        
        # Add common location variations
        location_variations = {
            'manila': ['ncr', 'metro manila', 'national capital region'],
            'cebu': ['cebu city', 'metro cebu'],
            'davao': ['davao city', 'davao del sur'],
            'cotabato': ['north cotabato'],
            'ilocos': ['ilocos norte', 'ilocos sur'],
            'leyte': ['southern leyte', 'northern leyte'],
            'samar': ['eastern samar', 'western samar', 'northern samar']
        }
        
        # Add variations if the location is a key in our variations dictionary
        for key, variations in location_variations.items():
            if key in location.lower():
                search_terms.extend(variations)
        
        # Remove duplicates while preserving order
        search_terms = list(dict.fromkeys(search_terms))
        
        # Create a mask for matches
        location_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        
        # Check each location column for matches
        for col in available_columns:
            # Check for matches in this column
            col_mask = df_filtered[col].apply(
                lambda x: self._is_location_match(x, search_terms)
            )
            location_mask = location_mask | col_mask
        
        # Also check the standardized location column if it exists
        if 'location_standard' in df_filtered.columns:
            std_mask = df_filtered['location_standard'].apply(
                lambda x: any(term in str(x).lower() for term in search_terms)
            )
            location_mask = location_mask | std_mask
        
        # Filter the dataframe
        result = df_filtered[location_mask].copy()
        
        # If we have coordinates, we can also do geospatial search
        if 'Latitude' in df_filtered.columns and 'Longitude' in df_filtered.columns:
            # This is a placeholder - you would implement actual geocoding here
            pass
            
        return result
    
    def _extract_years(self, query: str) -> tuple:
        """
        Extract year or year range from query.
        
        Args:
            query: The search query
            
        Returns:
            tuple: (start_year, end_year) - If a single year, both will be the same.
                   Returns (None, None) if no year is found.
        """
        # Look for year ranges (e.g., 2020-2023, 2020 to 2023)
        range_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4})',  # 2020-2023 or 2020 - 2023
            r'from\s+(\d{4})\s+to\s+(\d{4})',  # from 2020 to 2023
            r'between\s+(\d{4})\s+and\s+(\d{4})',  # between 2020 and 2023
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    start_year = int(match.group(1))
                    end_year = int(match.group(2))
                    return (min(start_year, end_year), max(start_year, end_year))
                except (ValueError, IndexError):
                    continue
        
        # Look for relative year references
        current_year = datetime.now().year
        relative_patterns = [
            (r'last\s+(\d+)\s+years?', lambda y: (current_year - y, current_year - 1)),
            (r'past\s+(\d+)\s+years?', lambda y: (current_year - y, current_year - 1)),
            (r'next\s+(\d+)\s+years?', lambda y: (current_year, current_year + y - 1)),
            (r'this\s+year', lambda _: (current_year, current_year)),
            (r'last\s+year', lambda _: (current_year - 1, current_year - 1)),
        ]
        
        for pattern, func in relative_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    if match.groups():
                        years = int(match.group(1))
                        return func(years)
                    return func(None)
                except (ValueError, IndexError):
                    continue
        
        # Look for a single year
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            try:
                year = int(year_match.group(1))
                return (year, year)
            except (ValueError, IndexError):
                pass
                
        return (None, None)

    def _filter_by_year(self, df: pd.DataFrame, year: int, date_range: tuple = None) -> pd.DataFrame:
        """
        Filter dataframe by year or date range with flexible handling.
        
        Args:
            df: DataFrame to filter
            year: Single year to filter by (deprecated, use date_range instead)
            date_range: Optional tuple of (start_year, end_year) for range filtering
            
        Returns:
            Filtered DataFrame containing only records within the specified year/range
        """
        if df.empty:
            return df
            
        # Create a copy to avoid SettingWithCopyWarning
        df_filtered = df.copy()
        
        # Define possible date/year column names (case insensitive)
        date_columns = [
            'CompletionDate', 'StartDate', 'Date', 'CompletionDate_parsed', 'StartDate_parsed'
        ]
        year_columns = [
            'CompletionYear', 'FundingYear', 'InfraYear', 'Year', 'ProjectYear', 'FiscalYear'
        ]
        
        # Find which columns exist in the dataframe
        existing_date_columns = [col for col in date_columns if col in df_filtered.columns]
        existing_year_columns = [col for col in year_columns if col in df_filtered.columns]
        
        # If no date/year columns found, return empty DataFrame
        if not existing_date_columns and not existing_year_columns:
            return pd.DataFrame()
        
        # Determine the year range to filter by
        if date_range and len(date_range) == 2:
            start_year, end_year = date_range
        elif year:
            start_year = end_year = year
        else:
            return df_filtered  # No year filter to apply
        
        # Create a mask for year matches
        year_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        
        # First, try to filter by date columns if available
        for col in existing_date_columns:
            # Skip if the column is all null
            if df_filtered[col].isnull().all():
                continue
                
            # Handle parsed datetime columns
            if col.endswith('_parsed'):
                # Filter by date range
                if start_year and end_year:
                    start_date = datetime(start_year, 1, 1)
                    end_date = datetime(end_year, 12, 31, 23, 59, 59)
                    col_mask = (df_filtered[col] >= start_date) & (df_filtered[col] <= end_date)
                    year_mask = year_mask | col_mask
            else:
                # Try to extract years from string dates
                year_series = df_filtered[col].astype(str).str.extract(r'(\d{4})')[0]
                year_series = pd.to_numeric(year_series, errors='coerce')
                if start_year == end_year:
                    col_mask = (year_series == start_year)
                else:
                    col_mask = (year_series >= start_year) & (year_series <= end_year)
                year_mask = year_mask | col_mask
        
        # If we didn't find any date columns or no matches, try year columns
        if not year_mask.any() and existing_year_columns:
            for col in existing_year_columns:
                # Convert to numeric, coercing errors to NaN
                year_series = pd.to_numeric(df_filtered[col], errors='coerce')
                
                # Filter by year range
                if start_year == end_year:
                    col_mask = (year_series == start_year)
                else:
                    col_mask = (year_series >= start_year) & (year_series <= end_year)
                
                year_mask = year_mask | col_mask
        
        # Apply the filter
        return df_filtered[year_mask] if year_mask.any() else pd.DataFrame()
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of query to determine the appropriate processing approach.
        
        Args:
            query: The user's query in lowercase
            
        Returns:
            str: The query type classification
        """
        # Extract location and year first, as they're common across many query types
        location = self._extract_location(query)
        year = self._extract_year(query)
        
        # Check for location or year-specific queries first
        has_location = location is not None
        has_year = year is not None
        
        # Location-specific queries (high priority)
        location_terms = [
            'in ', 'at ', 'location', 'region', 'province', 'city', 'municipality', 
            'barangay', 'where', 'near', 'around', 'area', 'district', 'zone'
        ]
        if has_location or any(term in query for term in location_terms):
            # If the query is specifically about locations (e.g., 'list locations', 'cities with projects')
            if any(term in query for term in ['list locations', 'cities with', 'municipalities in', 'provinces with']):
                return 'location_analysis'
            # If the query is about projects in a specific location
            elif has_location or any(term in query for term in ['projects in', 'flood control in', 'drainage in']):
                return 'location_analysis'
        
        # Year-specific queries (high priority)
        year_terms = ['in 20', 'year ', 'completed in', 'started in', 'funded in', 'during ', 'since ', 'from ']
        if has_year or any(term in query for term in year_terms):
            if any(term in query for term in ['completed in', 'finished in', 'delivered in']):
                return 'completion_analysis'
            elif any(term in query for term in ['started in', 'begun in', 'initiated in']):
                return 'completion_analysis'
            elif has_year or any(term in query for term in ['in 20', 'during ', 'since ', 'from ']):
                return 'analysis_insights'
        
        # Metadata/system queries (high priority)
        if any(term in query for term in ['how many total', 'total projects', 'dataset', 'unique values', 'missing data', 'columns', 'fields']):
            return 'metadata_analysis'
        
        # Analysis/insights queries (high priority)  
        if any(term in query for term in ['distribution', 'trend', 'growth', 'statistics', 'insights', 'analyze', 'analysis']):
            return 'analysis_insights'
            
        # Cost-related queries (high priority)
        cost_terms = [
            'cost', 'price', 'budget', 'expensive', 'cheap', 'affordable', 
            'funding', 'allocation', 'amount', '₱', 'PHP', 'pesos', 'million', 'billion'
        ]
        if any(term in query for term in cost_terms):
            return 'cost_analysis'
            
        # Contractor-related queries
        contractor_terms = [
            'contractor', 'company', 'firm', 'awarded to', 'implemented by', 
            'constructed by', 'built by', 'contractor name', 'who built', 'who constructed'
        ]
        if any(term in query for term in contractor_terms):
            return 'contractor_analysis'
            
        # Completion status queries
        status_terms = [
            'completed', 'ongoing', 'in progress', 'status', 'progress', 
            'finished', 'delivered', 'implemented', 'current state', 'phase', 'stage'
        ]
        time_terms = [
            'this year', 'last year', 'recent', 'latest', 'newest', 'oldest',
            'recently', 'when was', 'date', 'timeframe', 'schedule'
        ]
        if any(term in query for term in status_terms) or any(term in query for term in time_terms):
            return 'completion_analysis'
        
        # Project type queries
        type_terms = [
            'type', 'kind', 'drainage', 'bridge', 'seawall', 'revetment', 
            'mitigation', 'flood control', 'pumping', 'embankment', 'dike',
            'canal', 'culvert', 'slope protection', 'river control',
            'category', 'categories', 'classify', 'types of', 'kinds of',
            'what kind', 'what type', 'classification', 'project category'
        ]
        if any(term in query for term in type_terms):
            return 'project_type_analysis'
            
        # Comparison queries
        comparison_terms = [
            'compare', 'vs', 'versus', 'difference', 'similar', 'contrast', 
            'compared to', 'versus', 'versus.', 'vs.', 'difference between',
            'which is better', 'which one', 'versus others', 'versus other'
        ]
        if any(term in query for term in comparison_terms):
            return 'comparison'
            
        # If we have a location but no other clear classification
        if has_location:
            return 'location_analysis'
            
        # If we have a year but no other clear classification
        if has_year:
            return 'analysis_insights'
            
        # Default to general search
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
        """
        Perform semantic search using TF-IDF with improved result quality.
        
        Args:
            df: Filtered DataFrame to search within
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of matching records with similarity scores
        """
        if df.empty:
            return []
            
        try:
            # Get indices of the filtered dataframe in the original dataframe
            filtered_indices = df.index.tolist()
            
            # If no records after filtering, return empty list
            if not filtered_indices:
                return []
            
            # Transform query to TF-IDF
            query_vec = self.vectorizer.transform([query])
            
            # Calculate cosine similarity between query and documents
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix[filtered_indices]).flatten()
            
            # Get top k most similar documents, but only if they have some similarity
            similarity_threshold = 0.05  # Minimum similarity score to include
            
            # Create a list of (index, score) pairs and filter by threshold
            scored_docs = [(i, score) for i, score in enumerate(cosine_similarities) 
                          if score > similarity_threshold]
            
            # Sort by score in descending order
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k results
            top_indices = [i for i, _ in scored_docs[:top_k]]
            
            # Convert to list of records with similarity scores
            results = []
            for i in top_indices:
                record = df.iloc[i].to_dict()
                record['_similarity'] = float(cosine_similarities[i])
                results.append(record)
            
            # If no results above threshold, return the top result regardless
            if not results and len(cosine_similarities) > 0:
                best_idx = cosine_similarities.argmax()
                if cosine_similarities[best_idx] > 0:  # Only include if some similarity exists
                    record = df.iloc[best_idx].to_dict()
                    record['_similarity'] = float(cosine_similarities[best_idx])
                    results.append(record)
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            # Fall back to simple text search if semantic search fails
            return self._fallback_text_search(df, query, top_k)
    
    def _fallback_text_search(self, df: pd.DataFrame, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback text search when semantic search fails.
        Performs a simple case-insensitive text search across all string columns.
        """
        try:
            if df.empty:
                return []
                
            # Create a mask for rows containing the query
            mask = pd.Series(False, index=df.index)
            
            # Search in all string columns
            for col in df.select_dtypes(include=['object']):
                mask = mask | df[col].astype(str).str.lower().str.contains(query.lower(), na=False)
            
            # Get matching rows
            results = df[mask].head(top_k).to_dict('records')
            
            # Add a default similarity score
            for record in results:
                record['_similarity'] = 1.0
                
            return results
            
        except Exception as e:
            print(f"Error in fallback text search: {str(e)}")
            return []
    
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
