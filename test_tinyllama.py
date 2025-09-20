import sys
import time
from llm_handler import LLMHandler

def test_tinyllama():
    print("Testing TinyLlama with your application...")
    
    # Initialize with local model preference
    print("Initializing LLM handler with TinyLlama...")
    start_time = time.time()
    
    try:
        # Force using local model (TinyLlama)
        handler = LLMHandler(prefer_local=True)
        
        if not hasattr(handler, 'client') or handler.client is None:
            print("Error: Failed to initialize TinyLlama model")
            print("\nMake sure you have installed the required dependencies:")
            print("pip install -r requirements.txt")
            return
            
        init_time = time.time() - start_time
        print(f"Model loaded in {init_time:.2f} seconds")
        
        # Test general conversation
        print("\nTesting general conversation...")
        start_time = time.time()
        response = handler.generate_response("Hello, can you tell me about yourself?")
        gen_time = time.time() - start_time
        print(f"Response (generated in {gen_time:.2f}s):\n{response}")
        
        # Test flood control query
        print("\nTesting flood control query...")
        test_records = [
            {
                "ProjectName": "Cebu City Flood Control",
                "Location": "Cebu City",
                "Cost": "15000000",
                "Contractor": "ABC Construction",
                "Status": "Completed"
            },
            {
                "ProjectName": "Mandaue Drainage System",
                "Location": "Mandaue City",
                "Cost": "25000000",
                "Contractor": "XYZ Builders",
                "Status": "Ongoing"
            }
        ]
        
        start_time = time.time()
        response = handler.generate_response(
            "What flood control projects are there in Cebu?",
            records=test_records
        )
        gen_time = time.time() - start_time
        print(f"\nResponse (generated in {gen_time:.2f}s):\n{response}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Note: The first run will download the model (about 2.2GB).")
    print("This may take a few minutes depending on your internet connection.\n")
    test_tinyllama()
