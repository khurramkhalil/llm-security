import os
import logging
from logging_config import setup_logging

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


from attention_breaker.optim_layer_ranking import layer_ranking
from attention_breaker.q_learning_new import find_critical_bits
# from attention_breaker.weight_subset_selection import weight_subset_selection
# from attention_breaker.genbfa_optimization import genetic_optimization
# from attention_breaker.rl_bfa import find_critical_bits
# from attention_breaker.run_mmlu import main_

# Set up logging
setup_logging()

# Define the quantization configurTrue
quant_config = BitsAndBytesConfig(
    load_in_4bit=False,  # Set to True for 4-bit quantization
    load_in_8bit=True,  # Set to False for 8-bit quantization
    # llm_int8_threshold=6.0,  # Optional: threshold for mixed-precision
    llm_int8_skip_modules=None  # Optional: modules to skip for mixed-precision
)

# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Llama-3.1-8B" 
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
tokenizer.padding_side = 'left' 
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map=device)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.float16)

# Freeze the model's parameters (make it immutable)
for param in model.parameters():
    param.requires_grad = False

def main():
    alpha = 0.5
    top_k = 1000

    logging.info("Begin execution in main script")
    
    # Get layer sensitivity ranking
    sensitivity_losses = layer_ranking(model, tokenizer, alpha, top_k)

    # Find critical bits
    results = find_critical_bits(
        sensitivity_losses=sensitivity_losses,
        model=model,
        tokenizer=tokenizer,
        alpha=alpha,
        top_k=top_k
    )

    # Log results
    logging.info(f"Final model performance: {results['final_performance']}")
    logging.info(f"Most sensitive layers: {results['sensitivity_losses'][:5]}")

    # Access results
    print(f"Final model performance: {results['final_performance']}")
    print(f"Most sensitive layers: {results['sensitivity_losses'][:5]}")



if __name__ == "__main__":
    main()