import copy
import os
import random
import numpy as np
import torch

import pdb
from .mock_model import MockModel
from .eval_model import mmlu_evaluate, batch_mmlu_evaluate
from .run_mmlu import main_ as batch_mmlu_evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_gpu(weights):
    """GPU version of normalize function that uses L2 norm"""
    norm = torch.linalg.norm(weights.to(torch.float16))
    return weights / norm if norm != 0 else weights

def sscore_gpu(weights, gradients, alpha):
    """GPU version of sscore function"""
    wn = normalize_gpu(weights)
    grad_wn = normalize_gpu(gradients)
    return alpha * grad_wn + (1 - alpha) * wn

def bflip_gpu(weights, pos, indices):
    """GPU version of bit flip function"""
    result = weights.clone()
    indices = indices.to(torch.long)
    weights_int8 = result[indices].to(torch.int8)
    bit_mask = torch.tensor(1 << pos, device=weights.device, dtype=torch.int8)
    flipped_values = weights_int8 ^ bit_mask
    result[indices] = flipped_values.to(weights.dtype)
    return result

def swap_model_weights(model, layer, alpha, subsample_rate):
    model.eval()
    # device = next(model.parameters()).device  # Get model's device
    
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
            # k = int(subsample_rate * w1.numel() / (w1.numel() / 10))
            
            # Calculate scores
            scores = sscore_gpu(wf1, gradients, alpha)
            
            # Get top k indices on GPU
            top_k_indices = torch.argsort(scores, descending=True)[:subsample_rate]
            
            # Apply bit flipping
            perturbed_weights = bflip_gpu(wf1, 0, top_k_indices)
            
            # Reshape and assign back to parameter
            param.data = perturbed_weights.reshape(w1.shape)
            
    return model, top_k_indices

def layer_ranking(immutable_model, tokenizer, alpha, subsample_rate):
    sensitivity_losses = []

    # Get the dictionary of layers
    # all_layer_names = [name for name, module in immutable_model.named_modules() if list(module.parameters())]
    all_layer_names = [name for name, param in immutable_model.named_parameters()]
    original_acc = batch_mmlu_evaluate(immutable_model, tokenizer)
    print(f"########### Original Accuracy : {original_acc} ################################################")
    
    for layer in all_layer_names:
        # print(layer, param.shape)
        model = copy.deepcopy(immutable_model)
        # model.load_state_dict(immutable_model.state_dict())
        model, top_k_indices = swap_model_weights(model, layer, alpha, subsample_rate)
        
        # acc = mmlu_evaluate(model, tokenizer)
        acc = batch_mmlu_evaluate(model, tokenizer)
        sensitivity_losses.append((layer, acc, top_k_indices))
        # pdb.set_trace()
        print("######################################################################################################")
        print(f"Accuracy : {acc} , with layer: {layer}")
    
    # [param for name, param in model.named_parameters() if name==layer][0].flatten()[top_k_indices] 40159695   model.embed_tokens.weight  [0.1496    1]
    # sensitivity_losses.sort(key=lambda x: x[1], reverse=True)
    sensitivity_losses.sort(key=lambda x: x[1])
    return sensitivity_losses