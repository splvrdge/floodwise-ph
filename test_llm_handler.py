import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_handler import LLMHandler

def test_llm_handler():
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
        
    # Initialize the LLM handler
    handler = LLMHandler()
    
    # Test general conversation
    print("Testing general conversation...")
    try:
        response = handler.generate_response("Hello, how are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in general conversation test: {e}")
    
    # Test flood control query
    print("\nTesting flood control query...")
    test_records = [
        {
            "ProjectName": "Flood Control Project in Cebu City",
            "Location": "Cebu City",
            "Cost": "1000000",
            "Contractor": "ABC Construction",
            "Status": "Completed"
        },
        {
            "ProjectName": "Drainage System in Mandaue",
            "Location": "Mandaue City",
            "Cost": "2500000",
            "Contractor": "XYZ Builders",
            "Status": "Ongoing"
        }
    ]
    try:
        response = handler.generate_response("What flood control projects are there in Cebu?", test_records)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in flood control query test: {e}")

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Run the test
    test_llm_handler()
