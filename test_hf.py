import sys
import os
from hf_handler import HFModelHandler

def test_hf_model():
    """Test the Hugging Face model handler."""
    print("Testing Hugging Face model...")
    
    # Initialize the model handler with a smaller model for testing
    model_name = "microsoft/phi-2"  # A small but capable model
    print(f"Loading model: {model_name}")
    
    try:
        handler = HFModelHandler(model_name=model_name)
        print("Model loaded successfully!")
        
        # Test general conversation
        print("\nTesting general conversation...")
        response = handler.generate_response("Hello, how are you?")
        print(f"Response: {response}")
        
        # Test flood control query
        print("\nTesting flood control query...")
        response = handler.generate_response("Tell me about flood control projects in Cebu.")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    test_hf_model()
