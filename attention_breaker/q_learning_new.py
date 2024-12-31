import copy

import gymnasium as gym  # Using gymnasium instead of gym
from gymnasium import spaces
import numpy as np
import torch
import wandb
import json
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple, Optional
import logging

from .eval_model import mmlu_evaluate
from attention_breaker.optim_layer_ranking import bflip_gpu

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
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
    """Enhanced WandB callback with detailed logging of model performance and attack metrics"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode = 0
        
    def _on_step(self) -> bool:
        # Get the actual environment from the wrapped env
        env = self.training_env.envs[0].unwrapped
        
        # Log every step metrics
        wandb.log({
            'current_step': self.num_timesteps,
            'step_reward': self.locals.get('reward', 0),
            'llm_accuracy': env.current_performance,
            'performance_drop': env.original_performance - env.current_performance,
            'step_within_episode': env.steps
        })
        
        # Log action details
        last_info = self.locals.get('infos', [{}])[-1]
        if last_info:
            wandb.log({
                'attacked_layer': last_info.get('layer', ''),
                'weight_index': last_info.get('weight_idx', 0),
                'bit_position': last_info.get('bit_pos', 0)
            })
        
        if self.locals.get('done'):
            # Handle episode completion
            episode_reward = sum(self.episode_rewards)
            self.current_episode += 1
            
            # Calculate average episode reward safely
            n_steps = len(self.episode_rewards)
            avg_episode_reward = episode_reward / n_steps if n_steps > 0 else 0
            
            # Log episode-level metrics
            wandb.log({
                'episode': self.current_episode,
                'episode_reward': episode_reward,
                'episode_length': n_steps,
                'final_llm_accuracy': env.current_performance,
                'total_performance_drop': env.original_performance - env.current_performance,
                'average_episode_reward': avg_episode_reward
            })
            
            # Print detailed progress
            logger.info(f"\nEpisode {self.current_episode} Summary:")
            logger.info(f"Total Steps: {self.num_timesteps}")
            logger.info(f"Episode Reward: {episode_reward:.4f}")
            logger.info(f"Average Step Reward: {avg_episode_reward:.4f}")
            logger.info(f"LLM Accuracy: {env.current_performance:.4f}")
            logger.info(f"Performance Drop: {(env.original_performance - env.current_performance):.4f}")
            logger.info(f"Episode Length: {n_steps}")
            logger.info("-" * 50)
            
            self.episode_rewards = []
        else:
            self.episode_rewards.append(self.locals.get('reward', 0))
            
        return True

    def _on_training_start(self) -> None:
        """Log initial model performance"""
        env = self.training_env.envs[0].unwrapped
        wandb.log({
            'initial_llm_accuracy': env.original_performance,
            'training_start_time': wandb.run.start_time
        })
        logger.info(f"\nInitial LLM Accuracy: {env.original_performance:.4f}")
        logger.info("Starting training...\n")
        
    def _on_training_end(self) -> None:
        """Log final training statistics"""
        env = self.training_env.envs[0].unwrapped
        wandb.log({
            'final_llm_accuracy': env.current_performance,
            'total_episodes': self.current_episode,
            'total_steps': self.num_timesteps,
            'training_duration': wandb.run.duration
        })
        logger.info("\nTraining Complete!")
        logger.info(f"Final LLM Accuracy: {env.current_performance:.4f}")
        logger.info(f"Total Performance Drop: {(env.original_performance - env.current_performance):.4f}")
        logger.info(f"Total Episodes: {self.current_episode}")
        logger.info(f"Total Steps: {self.num_timesteps}")

class BitFlipEnv(gym.Env):
    def __init__(self, 
                 model: torch.nn.Module,
                 tokenizer,
                 sensitive_layers: List[Tuple[str, float, float]],
                 max_steps: int = 50,
                 history_size: int = 3,
                 top_k: int = 1000,
                 exploration_rate: float = 0.05):
        super().__init__()
        
        self._current_performance = None
        self._original_performance = None
        self.top_k = top_k
        self.exploration_rate = exploration_rate

        self.immutable_model = copy.deepcopy(model)
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.history_size = history_size
        self.steps = 0
        
        # Process sensitive layers (top 5)
        self.sensitive_layers = sensitive_layers[:5]
        self.layer_stats = self._compute_layer_stats()
        
        # Action space: [layer_idx, index_selection]
        self.action_space = spaces.MultiDiscrete([
            len(self.sensitive_layers),  # Layer selection
            self.top_k  # Index selection from top-k
        ])
        
        # Define observation space
        num_layers = len(self.sensitive_layers)
        state_length = 4 * num_layers + 1 + self.history_size * 2  # 4 stats per layer + current performance + action history

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_length,),
            dtype=np.float32
        )
        
        # Initialize performance tracking
        self.action_history = []
        self._original_performance = 0.71
        self._current_performance = self._original_performance
        
    @property
    def current_performance(self):
        """Make current performance accessible through property"""
        return self._current_performance
    
    @current_performance.setter
    def current_performance(self, value):
        self._current_performance = value
    
    @property
    def original_performance(self):
        """Make original performance accessible through property"""
        return self._original_performance
            


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state with seed support"""
        super().reset(seed=seed)  # Initialize self.np_random
        
        # Restore original weights
        self.model = self.immutable_model            
        self.steps = 0
        self.current_performance = self.original_performance
        self.action_history = []
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state, {}  # Return state and empty info dict

    def _normalize_gpu(self, weights):
        norm = torch.linalg.norm(weights.to(torch.float16))
        return weights / norm if norm != 0 else weights

    def _sscore_gpu(self, weights, gradients, alpha=0.5):
        wn = self._normalize_gpu(weights)
        grad_wn = self._normalize_gpu(gradients)
        return alpha * grad_wn + (1 - alpha) * wn

    def _compute_layer_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for layer_name, _, _ in self.sensitive_layers:
            param = dict(self.model.named_parameters())[layer_name]
            param_float = param.to(torch.float32)
            stats[layer_name] = {
                'mean': param_float.mean().item(),
                'std': param_float.std().item(),
                'max': param_float.max().item(),
                'min': param_float.min().item(),
                'size': param_float.numel()
            }
        return stats

    def step(self, action):
        self.steps += 1
        
        layer_idx, index_selection = action
        layer_name = self.sensitive_layers[layer_idx][0]
        
        # Get actual weight index from pre-computed sensitive indices
        weight_idx = self.sensitive_layers[layer_idx][2][index_selection]
        
        # Always flip bit at position 0
        bit_pos = 0
        
        # Perform bit flip
        self.model = copy.deepcopy(self.immutable_model)
        self.model = self._flip_bit(self.model, layer_name, weight_idx, bit_pos)
        
        # Evaluate new performance
        new_performance = self.evaluate_model(self.model)
        performance_drop = self.original_performance - new_performance
        
        # Calculate reward
        reward = performance_drop
        
        self.current_performance = new_performance
        self.action_history.append((layer_idx, index_selection))
        
        # Determine termination
        terminated = (self.current_performance < 0.2) or (self.steps >= self.max_steps)
        truncated = False
        
        return self._get_state(), reward, terminated, truncated, {
            'layer': layer_name,
            'weight_idx': weight_idx,
            'performance_drop': performance_drop
        }
    
    def _flip_bit(self, model: torch.nn.Module, layer_name: str, weight_idx: int, bit_pos: int):
        """Safe bit flipping operation with type checking"""
        model.eval()

        for name, param in model.named_parameters():
            if name in [layer_name]:
                
                # Keep weights on GPU
                w1 = param.data
                    
                print("Bit Position Index: ", weight_idx, "Layer:", layer_name)
                # Flatten weights while keeping on GPU
                wf1 = w1.flatten()
                
                # Apply bit flipping
                perturbed_weights = bflip_gpu(wf1, 0, weight_idx)
                
                # Reshape and assign back to parameter
                param.data = perturbed_weights.reshape(w1.shape)
                
        return model

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state with seed support"""
        super().reset(seed=seed)  # Initialize self.np_random
        
        # Restore original weights
        self.model = self.immutable_model            
        self.steps = 0
        self.current_performance = self.original_performance
        self.action_history = []
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state, {}  # Return state and empty info dict
    
    def _get_state(self) -> np.ndarray:
        # Compute layer statistics
        layer_stats = []
        for layer_name, _, _ in self.sensitive_layers:
            stats = self.layer_stats[layer_name]
            layer_stats.extend([
                stats['mean'],
                stats['std'],
                stats['max'],
                stats['min']
            ])
        
        # Pad action history if needed
        padded_history = np.zeros((self.history_size, 2))  # Assuming each action is a tuple of two elements
        for i, action in enumerate(self.action_history[-self.history_size:]):
            if i < len(self.action_history):
                padded_history[i] = action
        
        # Combine all state components
        state = np.concatenate([
            np.array(layer_stats),
            np.array([self.current_performance]),
            padded_history.flatten()
        ])
        
        return state.astype(np.float32)

    def evaluate_model(self, model) -> float:
        """Evaluate model performance"""
        return mmlu_evaluate(model, self.tokenizer)



def train_bit_flip_agent(
    model: torch.nn.Module,
    tokenizer,
    sensitivity_losses: List[Tuple[str, float]],
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    batch_size: int = 64
) -> Tuple[PPO, Dict]:
    """Train RL agent with improved hyperparameters"""
    exp_name = f"bit_flip_attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize WandB
    wandb.init(
        project="bit_flip_attack",
        config={
            'total_timesteps': total_timesteps,
            'max_steps': 50,
            'history_size': 3
        }
    )
    
    env = BitFlipEnv(model, tokenizer, sensitivity_losses)
    
    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # Added entropy coefficient for better exploration
        clip_range=0.2,
        verbose=1
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
    
    # Get final results
    config = {
        'total_timesteps': total_timesteps,
        'sensitive_layers': [layer[0] for layer in sensitivity_losses[:5]],
        'original_performance': env.original_performance,
        'final_performance': env.current_performance
    }
    
    with open(f"{save_path}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Training completed. Model saved to {save_path}")
    
    return agent, config

def find_critical_bits(
    sensitivity_losses: List[Tuple[str, float]],
    model: torch.nn.Module,
    tokenizer,
    alpha: float = 0.5,
    subsample_rate: int = 10
) -> Dict:
    """Main function to find critical bits using RL"""
    
    # Get layer sensitivity ranking
    logger.info("Starting layer sensitivity analysis...")
    # sensitivity_losses = layer_ranking(model, tokenizer, alpha, subsample_rate)
    
    # Train RL agent
    agent, config = train_bit_flip_agent(
        model,
        tokenizer,
        sensitivity_losses,
        # exp_name=f"bit_flip_alpha{alpha}_sr{subsample_rate}"
    )
    
    # Evaluate final performance
    env = BitFlipEnv(model, tokenizer, sensitivity_losses)
    final_performance = env.evaluate_model(model)
    
    results = {
        'config': config,
        'final_performance': final_performance,
        'sensitivity_losses': sensitivity_losses
    }
    
    return results