import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


import numpy as np
# from attention_breaker.layer_ranking import layer_ranking
from attention_breaker.optim_layer_ranking import layer_ranking
from attention_breaker.weight_subset_selection import weight_subset_selection
from attention_breaker.genbfa_optimization import genetic_optimization
from attention_breaker.mock_model import MockModel

# Define the quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=False,  # Set to True for 4-bit quantization
    load_in_8bit=True,  # Set to False for 8-bit quantization
    llm_int8_threshold=6.0,  # Optional: threshold for mixed-precision
    llm_int8_skip_modules=None  # Optional: modules to skip for mixed-precision
)

# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", quantization_config=quant_config, device_map=device)


# model = model.to('cpu')

def test_layer_ranking():
    # model = MockModel()
    alpha = 0.5
    subsample_rate = 10
    sensitivity_losses = layer_ranking(model, tokenizer, alpha, subsample_rate)
    return sensitivity_losses

def test_weight_subset_selection():
    # model = MockModel()
    gradients = np.random.rand(100)
    alpha = 0.5
    sensitivity_losses = [(layer['name'], 0.1) for layer in model.layers]
    subsample_rates = [5, 10, 20]
    loss_threshold = 0.2
    top_n_layers = 1
    selected_weights = weight_subset_selection(model, gradients, alpha, sensitivity_losses, subsample_rates, loss_threshold, top_n_layers)
    return selected_weights

def test_genetic_optimization():
    # model = MockModel()
    weight_subset = [1, 0, 1, 1, 0]
    loss_threshold = 0.3
    max_generations = 10
    mutation_rate = 0.1
    best_solution = genetic_optimization(model, weight_subset, loss_threshold, max_generations, mutation_rate)

    return best_solution

if __name__ == '__main__':
    sensitivity_losses = test_layer_ranking()
    selected_weights = test_weight_subset_selection()
    best_solution = test_genetic_optimization()
    
    assert(len(sensitivity_losses) > 0)
    assert(selected_weights is not None)
    assert(best_solution is not None)