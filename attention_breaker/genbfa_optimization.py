import gc
import copy
import heapq
import torch
import random
import numpy as np

# from .eval_model import mmlu_evaluate
from .optim_layer_ranking import sscore_gpu, bflip_gpu
from attention_breaker.run_mmlu import main_ as batch_mmlu_evaluate


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
            perturbed_weights = bflip_gpu(wf1, 0, torch.tensor(top_k_indices))
            
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

def genetic_optimization_top(immutable_model, selected_weights, tokenizer, max_generations, mutation_rate):
    population = [mutate(selected_weights[2], mutation_rate) for _ in range(100)]
    layer = selected_weights[0]
    
    # Use a min-heap to keep track of the top 5 solutions
    top_solutions = []
    for _ in range(max_generations):
        for candidate in population:

            try:
                # Use no_grad to reduce memory usage
                with torch.no_grad():
                    model = copy.deepcopy(immutable_model)
                    model = swap_model_weights(model, layer, candidate)
                    
                    acc = batch_mmlu_evaluate(model, tokenizer)
                    print("######################################################################################################")
                    print(f"Genetic Optimization:::: Accuracy : {acc} , with Candidate:  {candidate}")
                    
                    # Add the current candidate to the heap
                    if len(top_solutions) < 5:
                        heapq.heappush(top_solutions, (acc, candidate))
                    else:
                        # If the heap is full, push the new candidate and pop the worst one
                        heapq.heappushpop(top_solutions, (acc, candidate))

                    # Explicitly delete the model and clear cache
                    del model
                    gc.collect()  # Call garbage collector
                    torch.cuda.empty_cache()
            except:
                # Explicitly delete the model and clear cache
                del model
                gc.collect()  # Call garbage collector
                torch.cuda.empty_cache()                
                
                continue

        # Use the best solution from the current generation to create the next population
        best_solution = min(top_solutions, key=lambda x: x[0])[1]
        population = [mutate(best_solution, mutation_rate) for _ in range(100)]
    
    # Extract only the candidates from the top solutions
    top_candidates = [solution for _, solution in sorted(top_solutions)]
    top_indices = min(top_solutions, key=lambda x: x[0])[1]
    top_acc = min(top_solutions, key=lambda x: x[0])[0]
    
    return top_candidates, top_indices, top_acc