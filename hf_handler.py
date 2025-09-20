import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional
import logging
import streamlit as st
from contextlib import nullcontext

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device: str):
    """Load the model with caching to prevent reloading on reruns"""
    logger.info(f"Loading model {model_name}...")
    
    # Set the appropriate torch dtype based on device
    torch_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            dtype=torch_dtype,  # Using the new dtype parameter
            trust_remote_code=True,
            low_cpu_mem_usage=True  # More efficient memory usage
        )
        
        if device != "cuda":
            model = model.to(device)
        
        # Create a text generation pipeline with TinyLlama-specific settings
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            device=0 if device == "cuda" else -1
        )
        
        return model, tokenizer, pipe
        
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        # Provide more specific error messages for common issues
        if "CUDA out of memory" in str(e):
            raise RuntimeError("CUDA out of memory. Try using a smaller model or running on CPU.")
        elif "No space left on device" in str(e):
            raise RuntimeError("Disk space is full. Please free up space and try again.")
        raise
    
    return model, tokenizer, pipe

class HFModelHandler:
    """Handler for Hugging Face language models."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", show_loading: bool = True):
        """
        Initialize the Hugging Face model handler.
        
        Args:
            model_name: Name of the Hugging Face model to use.
                       Default is "TinyLlama/TinyLlama-1.1B-Chat-v1.0" which is lightweight
            show_loading: Whether to show loading spinner (set to False if loading is handled by parent)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Load model with caching
        spinner = st.spinner("Loading AI model (this may take a minute)...") if show_loading else nullcontext()
        with spinner:
            try:
                self.model, self.tokenizer, self.pipeline = load_model(self.model_name, self.device)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    @st.cache_resource(show_spinner=False)
    def load_model(self):
        """Load the model with caching to prevent reloading on reruns"""
        if self.model is None or self.tokenizer is None:
            self._load_model_implementation()
        return self.model, self.tokenizer
        
    def _load_model_implementation(self):
        """Actual model loading implementation"""
        try:
            logger.info(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
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
