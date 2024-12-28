import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from attention_breaker.optim_layer_ranking import layer_ranking
from attention_breaker.weight_subset_selection import weight_subset_selection
from attention_breaker.genbfa_optimization import genetic_optimization

from .eval_model import mmlu_evaluate

class BitFlipEnv(gym.Env):
    def __init__(self, model, tokenizer, sensitivity_losses, top_k_indices):
        super(BitFlipEnv, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.original_performance = self.evaluate_model()
        
        # Use sensitivity analysis results
        self.sensitive_layers = sensitivity_losses[:5]  # Top 5 sensitive layers
        self.layer_indices = top_k_indices
        
        # Define action space
        # [layer_selection, weight_index_selection, bit_position]
        self.action_space = spaces.MultiDiscrete([
            len(self.sensitive_layers),  # Number of sensitive layers
            1000,  # Reduced weight index space based on top_k_indices
            8      # Bits per weight (assuming 8-bit quantization)
        ])
        
        # Define observation space
        # [current_layer_stats, performance_delta, previous_actions]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,),  # Adjust based on your state representation
            dtype=np.float32
        )
        
    def step(self, action):
        # Unpack action
        layer_idx, weight_idx, bit_pos = action
        
        # Get actual layer and weight index from sensitive layers
        target_layer = self.sensitive_layers[layer_idx][0]
        actual_weight_idx = self.layer_indices[target_layer][weight_idx]
        
        # Perform bit flip
        self._flip_bit(target_layer, actual_weight_idx, bit_pos)
        
        # Get new state
        new_state = self._get_state()
        
        # Calculate reward
        current_performance = self.evaluate_model()
        reward = self.original_performance - current_performance
        
        # Check if episode should end
        done = (current_performance < 0.2) or (self.steps > self.max_steps)
        
        return new_state, reward, done, {}
    
    def reset(self):
        # Reset model to original state
        self.model.load_state_dict(self.original_state_dict)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        # Create state representation using:
        # 1. Statistics from sensitive layers
        # 2. Current performance delta
        # 3. History of recent actions
        state = []
        
        # Add layer statistics
        for layer_name, _ in self.sensitive_layers:
            layer = dict(self.model.named_parameters())[layer_name]
            state.extend([
                layer.mean().item(),
                layer.std().item(),
                layer.max().item(),
                layer.min().item()
            ])
            
        # Add performance delta
        current_perf = self.evaluate_model()
        state.append(self.original_performance - current_perf)
        
        return np.array(state, dtype=np.float32)
    
    def evaluate_model(self):
        return mmlu_evaluate(self.model, self.tokenizer)

def train_heuristic_rl():
    # Initialize environment with pre-computed sensitivity information
    env = BitFlipEnv(model, tokenizer, sensitivity_losses, top_k_indices)
    
    # Initialize stable-baselines3 PPO agent
    from stable_baselines3 import PPO
    
    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        batch_size=64,
        verbose=1
    )
    
    # Train agent
    agent.learn(total_timesteps=10000)
    
    return agent

def find_critical_bits_heuristic(model, tokenizer):
    # First, run sensitivity analysis
    sensitivity_losses = layer_ranking(model, tokenizer, alpha=0.5, subsample_rate=10)
    
    # Get top-k indices for sensitive layers
    selected_weights = weight_subset_selection(
        model, tokenizer, alpha=0.5,
        sensitivity_losses=sensitivity_losses,
        subsample_rates=[5, 10, 20],
        loss_threshold=0.10,
        top_n_layers=5
    )
    
    # Train RL agent with heuristic guidance
    agent = train_heuristic_rl()
    
    # Use trained agent to find critical bits
    critical_bits = evaluate_agent(agent)
    
    return critical_bits