#!/usr/bin/env python3
"""
Test script for Philippines Flood Control Projects Chatbot
"""

import pandas as pd
from data_handler import FloodControlDataHandler
from llm_handler import LLMHandler
import os

def test_data_handler():
    """Test the data handler functionality."""
    print("🧪 Testing Data Handler...")
    
    # Initialize handler
    handler = FloodControlDataHandler()
    
    # Test with sample CSV
    try:
        df = pd.read_csv('sample_flood_projects.csv')
        handler.df = df
        handler.text_columns = df.select_dtypes(include=['object']).columns.tolist()
        handler._prepare_search_index()
        
        print("✅ CSV loading: PASSED")
        
        # Test search functionality
        results = handler.search_relevant_records("Manila flood projects", top_k=3)
        if results:
            print("✅ Search functionality: PASSED")
            print(f"   Found {len(results)} relevant records")
        else:
            print("❌ Search functionality: FAILED")
            
        # Test summary stats
        stats = handler.get_summary_stats()
        if stats and 'total_records' in stats:
            print("✅ Summary statistics: PASSED")
            print(f"   Total records: {stats['total_records']}")
        else:
            print("❌ Summary statistics: FAILED")
            
    except Exception as e:
        print(f"❌ Data handler test failed: {e}")
        return False
    
    return True

def test_llm_handler():
    """Test the LLM handler functionality."""
    print("\n🤖 Testing LLM Handler...")
    
    try:
        handler = LLMHandler()
        
        # Test initialization
        print(f"✅ LLM initialization: PASSED")
        print(f"   Available: {handler.is_available()}")
        
        # Test model info
        info = handler.get_model_info()
        if info:
            print("✅ Model info: PASSED")
            print(f"   Model: {info['model']}")
        
        # Test fallback response
        sample_records = [
            {
                'Project_Name': 'Test Project',
                'Location': 'Test Location',
                'Cost_PHP_Millions': 100.0
            }
        ]
        
        response = handler._fallback_response("test query", sample_records)
        if response and len(response) > 0:
            print("✅ Fallback response: PASSED")
        else:
            print("❌ Fallback response: FAILED")
            
    except Exception as e:
        print(f"❌ LLM handler test failed: {e}")
        return False
    
    return True

def test_sample_data():
    """Test the sample data file."""
    print("\n📊 Testing Sample Data...")
    
    try:
        if os.path.exists('sample_flood_projects.csv'):
            df = pd.read_csv('sample_flood_projects.csv')
            print("✅ Sample CSV file: FOUND")
            print(f"   Records: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check for required columns
            required_cols = ['Project_Name', 'Location', 'Contractor']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if not missing_cols:
                print("✅ Required columns: PRESENT")
            else:
                print(f"⚠️  Missing columns: {missing_cols}")
                
        else:
            print("❌ Sample CSV file: NOT FOUND")
            return False
            
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'openai', 'python-dotenv', 
        'numpy', 'scikit-learn', 'sentence-transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}: INSTALLED")
        except ImportError:
            print(f"❌ {package}: MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🌊 Philippines Flood Control Projects Chatbot - Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_dependencies,
        test_sample_data,
        test_data_handler,
        test_llm_handler
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests PASSED! Your application is ready to run.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
    else:
        print("❌ Some tests FAILED. Please check the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure sample_flood_projects.csv exists")
        print("3. Check file permissions")

if __name__ == "__main__":
    main()
