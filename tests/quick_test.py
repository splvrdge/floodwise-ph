#!/usr/bin/env python3
"""
Quick test to verify the application works with the actual dataset
"""

import pandas as pd
from data_handler import FloodControlDataHandler

def test_with_actual_data():
    """Test with the actual flood control dataset."""
    print("🧪 Testing with actual flood control dataset...")
    
    try:
        # Load the actual dataset
        df = pd.read_csv('Dataset/flood-control-projects-table_2025-09-20.csv')
        print(f"✅ Dataset loaded: {len(df)} records")
        print(f"✅ Columns: {len(df.columns)} columns")
        
        # Initialize data handler
        handler = FloodControlDataHandler()
        handler.df = df
        handler.text_columns = df.select_dtypes(include=['object']).columns.tolist()
        handler._prepare_search_index()
        
        print("✅ Data handler initialized successfully")
        
        # Test search functionality
        test_queries = [
            "Palawan flood control projects",
            "drainage system construction",
            "Region IV-B projects",
            "DPWH projects 2024"
        ]
        
        for query in test_queries:
            results = handler.search_relevant_records(query, top_k=3)
            print(f"✅ Query '{query}': Found {len(results)} results")
        
        # Test summary stats
        stats = handler.get_summary_stats()
        print(f"✅ Summary stats: {stats['total_records']} total records")
        
        print("\n🎉 All tests passed! The application is ready to use with your dataset.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_with_actual_data()
