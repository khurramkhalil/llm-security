# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import wandb
import json
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple
import logging

from attention_breaker.optim_layer_ranking import layer_ranking
from attention_breaker.weight_subset_selection import weight_subset_selection
from attention_breaker.genbfa_optimization import genetic_optimization

from .eval_model import mmlu_evaluate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bit_flip_attack.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Log episode rewards
        if self.locals.get('done'):
            episode_reward = sum(self.episode_rewards)
            wandb.log({
                'episode_reward': episode_reward,
                'model_performance': self.training_env.get_attr('current_performance')[0]
            })
            self.episode_rewards = []
        else:
            self.episode_rewards.append(self.locals.get('reward', 0))
        return True

class BitFlipEnv(gym.Env):
    def __init__(self, 
                 model: torch.nn.Module,
                 tokenizer,
                 sensitive_layers: List[Tuple[str, float]],
                 max_steps: int = 50):
        super(BitFlipEnv, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.steps = 0
        
        # Store original model state
        self.original_state_dict = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
        # Initialize performance tracking
        self.original_performance = self.evaluate_model()
        self.current_performance = self.original_performance
        logger.info(f"Initial model performance: {self.original_performance}")
        
        # Process sensitive layers (top 5)
        self.sensitive_layers = sensitive_layers[:5]
        self.layer_stats = self._compute_layer_stats()
        
        # Action space: [layer_idx, weight_idx_percentage, bit_position]
        self.action_space = spaces.MultiDiscrete([
            len(self.sensitive_layers),  # Number of sensitive layers
            100,  # Weight index as percentage of layer size
            32    # Bit positions (supporting different precisions)
        ])
        
        # Observation space: [layer_stats, performance_metrics, action_history]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),  # Adjusted based on our state representation
            dtype=np.float32
        )
        
        # Track actions for state representation
        self.action_history = []
        
    def _compute_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute and store relevant statistics for each sensitive layer"""
        stats = {}
        for layer_name, _ in self.sensitive_layers:
            param = dict(self.model.named_parameters())[layer_name]
            stats[layer_name] = {
                'mean': param.mean().item(),
                'std': param.std().item(),
                'max': param.max().item(),
                'min': param.min().item(),
                'size': param.numel()
            }
        return stats
    
    def _get_state(self) -> np.ndarray:
        """Create state representation"""
        state = []
        
        # Layer statistics
        for layer_name, _ in self.sensitive_layers:
            current_param = dict(self.model.named_parameters())[layer_name]
            original_stats = self.layer_stats[layer_name]
            
            # Add current stats and changes from original
            state.extend([
                current_param.mean().item(),
                current_param.std().item(),
                current_param.mean().item() - original_stats['mean'],
                current_param.std().item() - original_stats['std']
            ])
        
        # Performance metrics
        state.extend([
            self.current_performance,
            self.original_performance - self.current_performance,
            self.steps / self.max_steps
        ])
        
        # Recent actions (last 3)
        for action in self.action_history[-3:]:
            state.extend(action if action is not None else [0, 0, 0])
            
        return np.array(state, dtype=np.float32)
    
    def _flip_bit(self, layer_name: str, weight_idx: int, bit_pos: int):
        """Perform bit flip operation"""
        param = dict(self.model.named_parameters())[layer_name]
        weight_data = param.data.flatten()
        
        # Convert to int representation
        weight_int = weight_data[weight_idx].cpu().numpy().view(np.int32)
        
        # Flip the bit
        weight_int ^= (1 << bit_pos)
        
        # Convert back to float and update
        weight_data[weight_idx] = torch.tensor(
            weight_int.view(np.float32),
            device=param.device
        )
        param.data = weight_data.reshape(param.shape)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        
        # Process action
        layer_idx, weight_percent, bit_pos = action
        layer_name = self.sensitive_layers[layer_idx][0]
        
        # Convert percentage to actual index
        layer_size = self.layer_stats[layer_name]['size']
        weight_idx = int((weight_percent / 100) * layer_size)
        
        # Perform bit flip
        self._flip_bit(layer_name, weight_idx, bit_pos)
        
        # Update performance
        self.current_performance = self.evaluate_model()
        
        # Calculate reward
        performance_drop = self.original_performance - self.current_performance
        reward = performance_drop * (1 + (self.max_steps - self.steps) / self.max_steps)
        
        # Update action history
        self.action_history.append(action)
        if len(self.action_history) > 3:
            self.action_history.pop(0)
            
        # Check termination
        done = (self.current_performance < 0.2) or (self.steps >= self.max_steps)
        
        # Get new state
        new_state = self._get_state()
        
        # Log step information
        logger.info(f"Step {self.steps}: performance={self.current_performance:.4f}, reward={reward:.4f}")
        
        return new_state, reward, done, {
            'performance': self.current_performance,
            'layer': layer_name,
            'weight_idx': weight_idx,
            'bit_pos': bit_pos
        }
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Restore original weights
        for name, param in self.model.named_parameters():
            param.data.copy_(self.original_state_dict[name])
            
        self.steps = 0
        self.current_performance = self.original_performance
        self.action_history = []
        
        return self._get_state()
    
    def evaluate_model(self) -> float:
        """Evaluate model performance"""
        return mmlu_evaluate(self.model, self.tokenizer)

def train_bit_flip_agent(
    model: torch.nn.Module,
    tokenizer,
    sensitivity_losses: List[Tuple[str, float]],
    exp_name: str = None,
    total_timesteps: int = 100000
) -> Tuple[PPO, List[Dict]]:
    """Train RL agent for bit flip attack"""
    
    # Initialize WandB
    exp_name = exp_name or f"bit_flip_attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="bit_flip_attack", name=exp_name)
    
    # Create environment
    env = BitFlipEnv(model, tokenizer, sensitivity_losses)
    
    # Initialize PPO agent
    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"./runs/{exp_name}"
    )
    
    # Setup callback
    callback = WandbCallback()
    
    # Train agent
    logger.info("Starting training...")
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # Save agent
    save_path = f"./models/{exp_name}"
    os.makedirs(save_path, exist_ok=True)
    agent.save(f"{save_path}/final_model")
    
    # Save experiment config
    config = {
        'exp_name': exp_name,
        'total_timesteps': total_timesteps,
        'sensitive_layers': [layer[0] for layer in sensitivity_losses[:5]],
        'original_performance': env.original_performance
    }
    
    with open(f"{save_path}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Training completed. Model saved to {save_path}")
    
    return agent, config

def find_critical_bits(
    model: torch.nn.Module,
    tokenizer,
    alpha: float = 0.5,
    subsample_rate: int = 10
) -> Dict:
    """Main function to find critical bits using RL"""
    
    # Get layer sensitivity ranking
    logger.info("Starting layer sensitivity analysis...")
    sensitivity_losses = layer_ranking(model, tokenizer, alpha, subsample_rate)
    
    # Train RL agent
    agent, config = train_bit_flip_agent(
        model,
        tokenizer,
        sensitivity_losses,
        exp_name=f"bit_flip_alpha{alpha}_sr{subsample_rate}"
    )
    
    # Evaluate final performance
    env = BitFlipEnv(model, tokenizer, sensitivity_losses)
    final_performance = env.evaluate_model()
    
    results = {
        'config': config,
        'final_performance': final_performance,
        'sensitivity_losses': sensitivity_losses
    }
    
    return results