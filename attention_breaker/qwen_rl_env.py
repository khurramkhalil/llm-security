import gc
import os
import pdb
import csv
import copy
import random
import numpy as np

import struct


import torch
# from .eval_model import mmlu_evaluate, batch_mmlu_evaluate
from .run_mmlu import main_ as batch_mmlu_evaluate

import logging
import sys
sys.path.append('..')  # Add the parent directory to the system path
from logging_config import setup_logging

# Set up logging
setup_logging()


class BitFlipEnv:
    def __init__(self, model, tokenizer, layer_name, top_k_indices, max_flips=10):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        self.top_k_indices = top_k_indices
        self.max_flips = max_flips
        self.current_bits = self._get_initial_bits()
        self.flips_so_far = 0
        self.original_accuracy = batch_mmlu_evaluate(model, tokenizer)
    
    def _get_initial_bits(self):
        # Extract the weights corresponding to the top_k_indices
        param = [param for name, param in self.model.named_parameters() if name == self.layer_name][0]
        weights = param.data.flatten()[self.top_k_indices]
        # Convert to binary representation (for simplicity, assume float32)
        bits = []
        for weight in weights.cpu().numpy():
            bits.extend([int(b) for b in np.binary_repr(struct.unpack('I', struct.pack('f', weight))[0], width=32)])
        return bits
    
    def reset(self):
        self.current_bits = self._get_initial_bits()
        self.flips_so_far = 0
        return self.current_bits

    def step(self, action):
        # Flip the bit at the specified index
        self.current_bits[action] = 1 - self.current_bits[action]
        self.flips_so_far += 1
        
        # Update the model weights
        new_weights = self._bits_to_weights(self.current_bits)
        self._update_model_weights(new_weights)
        
        # Evaluate the model's performance
        new_accuracy = batch_mmlu_evaluate(self.model, self.tokenizer)
        reward = self.original_accuracy - new_accuracy  # Higher reward for more degradation
        
        done = self.flips_so_far >= self.max_flips
        
        return self.current_bits, reward, done

    def _bits_to_weights(self, bits):
        # Convert binary bits back to float32 weights
        weights = []
        for i in range(0, len(bits), 32):
            bit_str = ''.join(str(bit) for bit in bits[i:i+32])
            int_val = int(bit_str, 2)
            weight = struct.unpack('f', struct.pack('I', int_val))[0]
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)

    def _update_model_weights(self, new_weights):
        param = [param for name, param in self.model.named_parameters() if name == self.layer_name][0]
        original_shape = param.data.shape
        flat_param = param.data.flatten()
        flat_param[self.top_k_indices] = new_weights
        param.data = flat_param.reshape(original_shape)