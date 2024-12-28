import copy
import os
import random
import numpy as np
import torch

from .mock_model import MockModel
from .eval_model import mmlu_evaluate


def normalize(weights):
    norm = np.linalg.norm(weights)
    return weights / norm if norm != 0 else weights

def sscore(weights, gradients, alpha):
    wn = normalize(weights)
    grad_wn = normalize(gradients)
    return alpha * grad_wn + (1 - alpha) * wn

def bflip(weights, pos, indices):
    # original_weights = np.copy(weights)
    for i in indices:
        weights[i] = weights[i].astype(np.int8) ^ (1 << pos)
    return weights

def swap_model_weights(model, layer, param, alpha, subsample_rate):

    model.eval()
    for name, param in model.named_parameters():
        if name in [layer]: 
            print("Selected layer", name)
            print(param.shape)
            w1 = param.data
            wf1 = w1.detach().cpu().numpy().flatten()

            gradients = np.random.rand(wf1.shape[0])
            k = int(subsample_rate * len(wf1) / (len(wf1) / 10))
            scores = sscore(wf1, gradients, alpha)
            top_k_indices = np.argsort(scores)[-k:]
            perturbed_weights = bflip(wf1, 0, top_k_indices)

            wf1 = torch.tensor(perturbed_weights)
            
            wf11 = wf1.reshape(w1.shape)
            param.data = wf11

    return model

def layer_ranking(immutable_model, tokenizer, alpha, subsample_rate):
    sensitivity_losses = []

    # Get the dictionary of layers
    # all_layer_names = [name for name, module in immutable_model.named_modules() if list(module.parameters())]
    # all_layer_names = [name for name, param in immutable_model.named_parameters()]

    for layer, param in immutable_model.named_parameters():
        print(layer, param.shape)
        model = copy.deepcopy(immutable_model)
        model = swap_model_weights(model, layer, param, alpha, subsample_rate)

        loss = mmlu_evaluate(model, tokenizer)
        sensitivity_losses.append((layer, loss))
        print("######################################################################################################")
        print(f"Loss (1 - accuracy): {loss} with layer: {layer}")

    sensitivity_losses.sort(key=lambda x: x[1], reverse=True)
    return sensitivity_losses

def get_all_parameters(model):

    return {name: list(module.parameters()) for name, module in model.named_modules() if list(module.parameters())}

def get_layer_parameters(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return list(module.parameters())
    return None

def get_layers_with_parameters(model):
    layers_dict = {}
    
    for name, module in model.named_modules():
        # Check if module has parameters
        params = list(module.parameters())
        if len(params) > 0:  # Only include layers that have parameters
            # Get parameter info
            param_info = {
                'type': type(module).__name__,
                'parameters': {
                    'total_params': sum(p.numel() for p in params),
                    'param_shapes': {
                        name: tuple(p.shape) 
                        for name, p in module.named_parameters()
                    }
                }
            }
            layers_dict[name] = param_info
            
    return layers_dict