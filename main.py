import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


import numpy as np
from attention_breaker.optim_layer_ranking import layer_ranking
from attention_breaker.weight_subset_selection import weight_subset_selection
from attention_breaker.genbfa_optimization import genetic_optimization

# Define the quantization configurTrue
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Set to True for 4-bit quantization
    load_in_8bit=False,  # Set to False for 8-bit quantization
    llm_int8_threshold=6.0,  # Optional: threshold for mixed-precision
    llm_int8_skip_modules=None  # Optional: modules to skip for mixed-precision
)

# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", quantization_config=quant_config, device_map=device)




def main():
    # Define parameters
    alpha = 0.5
    subsample_rate = 10
    subsample_rates = [5, 10, 20]
    loss_threshold = 0.10
    top_n_layers = 2
    max_generations = 10
    mutation_rate = 0.1

    # Step 1: Layer Ranking
    print("Performing layer ranking...")
    sensitivity_losses = layer_ranking(model, tokenizer, alpha, subsample_rate)
    print("Layer ranking completed. Sensitive layers identified.")

    # Step 2: Weight Subset Selection
    print("Selecting weight subset...")
    selected_weights = weight_subset_selection(model, tokenizer, alpha, sensitivity_losses, subsample_rates, loss_threshold, top_n_layers)
    layer_name, weight_indices = selected_weights[0]
    print(f"Selected weights from layer: {layer_name}, indices: {weight_indices}")

    # Step 3: Genetic Optimization
    print("Optimizing weight subset...")

    best_solution = genetic_optimization(model, selected_weights[0], tokenizer,loss_threshold, max_generations, mutation_rate)
    print(f"Optimized weight subset: {best_solution}")

if __name__ == "__main__":
    main()