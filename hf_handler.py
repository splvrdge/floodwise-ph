import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFModelHandler:
    """Handler for Hugging Face language models."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the Hugging Face model handler.
        
        Args:
            model_name: Name of the Hugging Face model to use.
                       Default is "TinyLlama/TinyLlama-1.1B-Chat-v1.0" which is lightweight
                       and works well for 1-2 users.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Create a text generation pipeline with TinyLlama-specific settings
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            logger.info(f"Model {self.model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 500, **kwargs) -> str:
        """
        Generate a response using the Hugging Face model.
        
        Args:
            prompt: The input prompt for the model.
            max_length: Maximum length of the generated response.
            **kwargs: Additional arguments for text generation.
            
        Returns:
            Generated text response.
        """
        if not self.pipeline:
            return "Error: Model not initialized properly."
            
        try:
            # Prepare the prompt with instructions
            formatted_prompt = f"""You are a helpful AI assistant that provides information about flood control projects in the Philippines.
            
            User: {prompt}
            
            Assistant:"""
            
            # Generate response
            response = self.pipeline(
                formatted_prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Extract and clean the response
            full_text = response[0]['generated_text']
            # Remove the input prompt from the response
            generated_text = full_text.replace(formatted_prompt, "").strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if the model is available and ready to use."""
        return self.pipeline is not None


if __name__ == "__main__":
    # Example usage
    handler = HFModelHandler()
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
            
        response = handler.generate_response(user_input)
        print("\nAssistant:", response)
