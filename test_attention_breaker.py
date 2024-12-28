import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


import unittest
import numpy as np
from attention_breaker.layer_ranking import layer_ranking
from attention_breaker.weight_subset_selection import weight_subset_selection
from attention_breaker.genbfa_optimization import genetic_optimization
from attention_breaker.mock_model import MockModel


# # Loading the tokenizer and model from Hugging Face's model hub.
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# # using CUDA for an optimal experience
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)



class TestAttentionBreaker(unittest.TestCase):

    def test_layer_ranking(self):
        model = MockModel()
        gradients = np.random.rand(100)
        alpha = 0.5
        subsample_rate = 10
        sensitivity_losses = layer_ranking(model, gradients, alpha, subsample_rate)
        self.assertTrue(len(sensitivity_losses) > 0)

    def test_weight_subset_selection(self):
        model = MockModel()
        gradients = np.random.rand(100)
        alpha = 0.5
        sensitivity_losses = [(layer['name'], 0.1) for layer in model.layers]
        subsample_rates = [5, 10, 20]
        loss_threshold = 0.2
        top_n_layers = 1
        selected_weights = weight_subset_selection(model, gradients, alpha, sensitivity_losses, subsample_rates, loss_threshold, top_n_layers)
        self.assertTrue(selected_weights is not None)

    def test_genetic_optimization(self):
        model = MockModel()
        weight_subset = [1, 0, 1, 1, 0]
        loss_threshold = 0.3
        max_generations = 10
        mutation_rate = 0.1
        best_solution = genetic_optimization(model, weight_subset, loss_threshold, max_generations, mutation_rate)
        self.assertTrue(best_solution is not None)

if __name__ == '__main__':
    unittest.main()