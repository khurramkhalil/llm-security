import copy
import torch
import numpy as np

from .optim_layer_ranking import sscore_gpu, bflip_gpu
from .eval_model import mmlu_evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swap_model_weights(model, layer, alpha, r):
    model.eval()
    device = next(model.parameters()).device  # Get model's device
    
    for name, param in model.named_parameters():
        if name in [layer]:
            # print("Selected layer:", name)
            # print("Shape:", param.shape)
            
            # Keep weights on GPU
            w1 = param.data
            
            # Generate random gradients directly on GPU
            gradients = torch.rand(w1.numel(), device=device)
            
            # Flatten weights while keeping on GPU
            wf1 = w1.flatten()
            
            # Calculate k
            k = int(r * w1.numel() / (w1.numel() / 10))
            
            # Calculate scores
            scores = sscore_gpu(wf1, gradients, alpha)
            
            # Get top k indices on GPU
            top_k_indices = torch.argsort(scores, descending=True)[:k]
            
            # Apply bit flipping
            perturbed_weights = bflip_gpu(wf1, 0, top_k_indices)
            
            # Reshape and assign back to parameter
            param.data = perturbed_weights.reshape(w1.shape)
            
    return model, top_k_indices


def weight_subset_selection(immutable_model, tokenizer, alpha, sensitivity_losses, subsample_rates, loss_threshold, top_n_layers):
    selected_weights = []
    top_layers = sensitivity_losses[:top_n_layers]

    for layer_name, _ in top_layers:
        for r in subsample_rates:
            model = copy.deepcopy(immutable_model)
            
            model, top_k_indices = swap_model_weights(model, layer_name, alpha, r)

            acc = mmlu_evaluate(model, tokenizer)
            print("######################################################################################################")
            print(f"Weight Subset Selection:::: Accuracy : {acc} , with layer: {layer_name}, subsample rate: {r}")
            if acc <= loss_threshold:
                selected_weights.insert(0, (layer_name, top_k_indices))
                break
            else:
               selected_weights.append((layer_name, top_k_indices))

    selected_weights.sort(key=lambda x: len(x[1]))
    
    return selected_weights