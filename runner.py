from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import ssl
import certifi
import os

class DeepseekLocal:
    def __init__(
        self,
        model_name="deepseek-ai/deepseek-coder-1.3b-base",
        cache_dir=None
    ):
        self.model_name = model_name
        # Check if running on MacOS with Metal
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.use_quantization = False
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.use_quantization = True
        else:
            self.device = "cpu"
            self.use_quantization = False
            
        self.cache_dir = cache_dir
        
        # Fix SSL certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context
        
        self.setup_model()
    
    def setup_model(self):
        """Initialize the model with optional quantization"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        # Setup model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
        }
        
        # Device-specific configurations
        if self.device == "cuda" and self.use_quantization:
            print("Using 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs.update({
                "quantization_config": quantization_config,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            })
        else:
            # For MPS (MacOS) or CPU
            model_kwargs.update({
                "torch_dtype": torch.float32,
                "device_map": {"": self.device}
            })
        
        try:
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate(
        self,
        prompt,
        max_length=2048,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
        **kwargs
    ):
        """Generate response from the model"""
        try:
            # Encode prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Decode response
            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return responses[0] if num_return_sequences == 1 else responses
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Set environment variables for SSL certificate verification
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    
    # Initialize model
    deepseek = DeepseekLocal(
        model_name="deepseek-ai/deepseek-coder-1.3b-base",
        cache_dir="./model_cache"
    )
    
    # Example prompts
    prompts = [
        "Write a Python function to implement binary search.",
    ]
    
    # Generate responses
    for prompt in prompts:
        print("\nPrompt:", prompt)
        print("\nGenerating response...")
        
        response = deepseek.generate(
            prompt,
            max_length=2048,
            temperature=0.7,
            top_p=0.95
        )
        
        print("\nResponse:", response)
