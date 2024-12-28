import random
import copy
import torch
import numpy as np

from .optim_layer_ranking import sscore_gpu, bflip_gpu
from .eval_model import mmlu_evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swap_model_weights(model, layer, top_k_indices):
    model.eval()
    
    for name, param in model.named_parameters():
        if name in [layer]:
            # Keep weights on GPU
            w1 = param.data

            # Flatten weights while keeping on GPU
            wf1 = w1.flatten()
            
            # Apply bit flipping
            perturbed_weights = bflip_gpu(wf1, 0, top_k_indices)
            
            # Reshape and assign back to parameter
            param.data = perturbed_weights.reshape(w1.shape)
        
    return model



def mutate(weights, mutation_rate):
    return [w if random.random() > mutation_rate else 0 for w in weights]

def genetic_optimization(immutable_model, selected_weights, tokenizer, loss_threshold, max_generations, mutation_rate):
    population = [mutate(selected_weights[1], mutation_rate) for _ in range(100)]
    layer = selected_weights[0]
    best_solution = None
    best_loss = float('inf')
    for _ in range(max_generations):
        for candidate in population:

            model = copy.deepcopy(immutable_model)
            model = swap_model_weights(model, layer, candidate)
            
            loss = mmlu_evaluate(model, tokenizer)
            print("######################################################################################################")
            print(f"Genetic Optimization:::: Accuracy : {loss} , with Candidate:  {candidate}")
            if loss <= loss_threshold and len(candidate) < best_loss:
                best_solution = candidate
                best_loss = loss
        population = [mutate(best_solution, mutation_rate) for _ in range(100)]
    return best_solution