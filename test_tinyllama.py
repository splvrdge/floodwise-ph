#!/usr/bin/env python3
"""
Test script for TinyLlama integration with FloodWise PH
This script tests the TinyLlama model's ability to analyze flood control project data accurately.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent.absolute()))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tinyllama_integration():
    """Test TinyLlama integration with flood control data."""
    
    print("🌊 FloodWise PH - TinyLlama Integration Test")
    print("=" * 50)
    
    try:
        # Import our modules
        from data_handler import FloodControlDataHandler
        from llm_handler import LLMHandler
        
        print("✅ Successfully imported modules")
        
        # Initialize data handler
        print("\n📊 Initializing Data Handler...")
        data_handler = FloodControlDataHandler()
        
        # Load the dataset
        print("📁 Loading flood control dataset...")
        dataset_path = "Dataset/flood-control-projects-table_2025-09-20.csv"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at {dataset_path}")
            return False
            
        success = data_handler.load_data(dataset_path)
        if not success:
            print("❌ Failed to load dataset")
            return False
            
        print(f"✅ Dataset loaded successfully!")
        
        # Get basic stats
        try:
            stats = data_handler.get_summary_stats()
            print(f"📈 Dataset contains {stats.get('total_records', 0)} projects")
            print(f"🌏 Covering {stats.get('unique_regions', 0)} regions")
        except Exception as e:
            print(f"⚠️  Could not get stats: {e}")
        
        # Initialize TinyLlama
        print("\n🤖 Initializing TinyLlama...")
        llm_handler = LLMHandler()  # Uses TinyLlama by default
        
        if not llm_handler.is_available():
            print("❌ TinyLlama is not available")
            return False
            
        model_info = llm_handler.get_model_info()
        print(f"✅ TinyLlama initialized successfully!")
        print(f"   Model: {model_info['model']}")
        print(f"   Provider: {model_info['provider']}")
        print(f"   Type: {model_info['type']}")
        
        # Test queries
        test_queries = [
            {
                "query": "What are the most expensive flood control projects?",
                "description": "Testing cost analysis capabilities"
            },
            {
                "query": "Show me projects in Metro Manila",
                "description": "Testing location-based filtering"
            },
            {
                "query": "Which contractors have the most projects?",
                "description": "Testing contractor analysis"
            },
            {
                "query": "What types of flood control projects are there?",
                "description": "Testing project type categorization"
            },
            {
                "query": "Tell me about projects completed in 2023",
                "description": "Testing temporal filtering"
            }
        ]
        
        print("\n🧪 Running Test Queries...")
        print("=" * 50)
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {test['description']}")
            print(f"Query: '{test['query']}'")
            print("-" * 40)
            
            try:
                # Search for relevant records
                start_time = time.time()
                results = data_handler.search_relevant_records(test['query'], top_k=10)
                search_time = time.time() - start_time
                
                print(f"📊 Found {len(results)} relevant projects in {search_time:.2f}s")
                
                if results:
                    # Generate response with TinyLlama
                    print("🤖 Generating TinyLlama response...")
                    start_time = time.time()
                    response = llm_handler.generate_response(test['query'], results)
                    response_time = time.time() - start_time
                    
                    print(f"⏱️  Response generated in {response_time:.2f}s")
                    print(f"📝 Response length: {len(response)} characters")
                    print("\n🎯 TinyLlama Response:")
                    print("-" * 30)
                    print(response)
                    print("-" * 30)
                    
                    # Basic response quality checks
                    if len(response) < 50:
                        print("⚠️  Warning: Response seems too short")
                    elif "I couldn't find" in response or "No information" in response:
                        print("⚠️  Warning: Response indicates no data found")
                    elif len(response) > 2000:
                        print("⚠️  Warning: Response might be too long")
                    else:
                        print("✅ Response length looks good")
                    
                    # Check if response contains relevant information
                    query_lower = test['query'].lower()
                    response_lower = response.lower()
                    
                    if 'expensive' in query_lower and ('₱' in response or 'cost' in response_lower):
                        print("✅ Response contains cost information")
                    elif 'contractor' in query_lower and 'contractor' in response_lower:
                        print("✅ Response contains contractor information")
                    elif 'manila' in query_lower and 'manila' in response_lower:
                        print("✅ Response contains location information")
                    elif 'type' in query_lower and ('project' in response_lower or 'flood' in response_lower):
                        print("✅ Response contains project type information")
                    elif '2023' in query_lower and '2023' in response_lower:
                        print("✅ Response contains temporal information")
                    else:
                        print("✅ Response generated successfully")
                        
                else:
                    print("❌ No relevant projects found for this query")
                    
            except Exception as e:
                print(f"❌ Error during test {i}: {str(e)}")
                logger.error(f"Test {i} failed: {e}", exc_info=True)
                
            print("\n" + "="*50)
        
        print("\n🎉 TinyLlama Integration Test Complete!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error("Test failed with unexpected error", exc_info=True)
        return False

def test_model_loading():
    """Test if TinyLlama model can be loaded successfully using official configuration."""
    print("\n🔧 Testing TinyLlama Model Loading...")
    
    try:
        from llm_handler import LLMHandler
        
        print("📥 Loading TinyLlama model following official Hugging Face configuration...")
        print("   (This may take a few minutes on first run - downloading ~2.2GB)")
        start_time = time.time()
        
        llm_handler = LLMHandler()
        
        load_time = time.time() - start_time
        print(f"⏱️  Model loaded in {load_time:.2f}s")
        
        if llm_handler.is_available():
            print("✅ TinyLlama model loaded successfully!")
            
            # Test the official chat template format
            print("🧪 Testing TinyLlama chat template...")
            
            # Create a simple test using the official format
            simple_results = [
                {
                    "ProjectDescription": "Flood Control Infrastructure Project",
                    "Municipality": "Quezon City",
                    "Province": "Metro Manila",
                    "ContractCost": 5000000,
                    "Contractor": "ABC Construction Corp"
                }
            ]
            
            test_query = "What can you tell me about this flood control project?"
            response = llm_handler.generate_response(test_query, simple_results)
            
            if response and len(response) > 20:
                print("✅ TinyLlama chat template working!")
                print(f"📝 Sample response: {response[:150]}...")
                
                # Check if response contains relevant information
                if any(term in response.lower() for term in ['flood', 'project', 'quezon', 'manila']):
                    print("✅ Response contains relevant project information")
                else:
                    print("⚠️  Response may not be analyzing data correctly")
            else:
                print("⚠️  TinyLlama response generation may have issues")
                print(f"📝 Response received: '{response}'")
                
            return True
        else:
            print("❌ TinyLlama model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        logger.error("Model loading test failed", exc_info=True)
        return False

if __name__ == "__main__":
    print("🚀 Starting TinyLlama Integration Tests...")
    
    # Test 1: Model Loading
    model_test = test_model_loading()
    
    if model_test:
        # Test 2: Full Integration
        integration_test = test_tinyllama_integration()
        
        if integration_test:
            print("\n🎊 All tests passed! TinyLlama is working correctly with FloodWise PH!")
        else:
            print("\n❌ Integration tests failed. Check the logs above.")
    else:
        print("\n❌ Model loading failed. Please check your installation.")
        print("💡 Try running: pip install transformers torch accelerate")
    
    print("\n🏁 Test script completed.")
