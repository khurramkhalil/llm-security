import torch
from typing import List, Tuple, Dict
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

class BitFlipEnv(gym.Env):
    def __init__(self, 
                 model: torch.nn.Module,
                 tokenizer,
                 sensitive_layers: List[Tuple[str, float]],
                 max_steps: int = 50,
                 history_size: int = 3,
                 batch_size: int = 16):  # Added batch_size parameter
        super().__init__()
        # ... existing initialization code ...
        self.batch_size = batch_size
        self.performance_history = []  # Track performance changes
        
    def calculate_reward(self, new_performance: float) -> float:
        """
        Calculate reward with improved logic:
        1. Consider relative performance drop
        2. Add momentum based on recent improvements
        3. Include exploration bonus for new territories
        4. Penalize oscillating behavior
        """
        performance_drop = self.current_performance - new_performance
        relative_drop = performance_drop / self.original_performance
        
        # Calculate momentum (trend in last few steps)
        self.performance_history.append(new_performance)
        if len(self.performance_history) > 5:
            self.performance_history.pop(0)
        
        # Calculate trend
        if len(self.performance_history) >= 2:
            trend = (self.performance_history[-1] - self.performance_history[0]) / len(self.performance_history)
        else:
            trend = 0
            
        # Exploration bonus for finding new low performances
        min_seen = min(self.performance_history) if self.performance_history else self.original_performance
        exploration_bonus = 0.1 if new_performance < min_seen else 0
        
        # Penalize oscillating behavior
        oscillation_penalty = 0
        if len(self.performance_history) >= 3:
            variations = np.diff(self.performance_history[-3:])
            if np.sign(variations[-1]) != np.sign(variations[-2]):
                oscillation_penalty = 0.05
        
        # Combine components
        base_reward = relative_drop * (1 + np.exp(-self.steps / (self.max_steps * 0.7)))
        momentum_factor = 0.2 * trend
        total_reward = (base_reward + momentum_factor + exploration_bonus - oscillation_penalty)
        
        return float(total_reward)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        
        layer_idx, weight_percent, bit_pos = action
        layer_name = self.sensitive_layers[layer_idx][0]
        
        # Convert percentage to actual index
        layer_size = self.layer_stats[layer_name]['size']
        weight_idx = int((weight_percent / 100) * layer_size)
        
        # Perform bit flip
        self._flip_bit(layer_name, weight_idx, bit_pos)
        
        # Update performance and calculate reward
        new_performance = self.evaluate_model()
        reward = self.calculate_reward(new_performance)
        
        self.current_performance = new_performance
        self.action_history.append(action)
        
        # Determine termination conditions
        terminated = (self.current_performance < 0.2) or (self.steps >= self.max_steps)
        truncated = False
        
        return self._get_state(), reward, terminated, truncated, {
            'performance': self.current_performance,
            'layer': layer_name,
            'weight_idx': weight_idx,
            'bit_pos': bit_pos,
            'reward_components': {
                'base_reward': reward,
                'current_performance': new_performance,
                'performance_history': self.performance_history.copy()
            }
        }

def batch_mmlu_evaluate(model, tokenizer, batch_size=16, num_samples=100):
    """Evaluate MMLU with batched processing for better GPU utilization"""
    dataset = load_dataset("cais/mmlu", "all")["validation"]
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    total_correct = 0
    total_samples = 0
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        
        # Prepare batch inputs
        prompts = [
            format_mmlu_prompt(item['question'], item['choices'])
            for item in batch_data
        ]
        
        # Tokenize all prompts
        encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate answers in batch
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0
            )
        
        # Process responses
        for j, output in enumerate(outputs):
            response = tokenizer.decode(
                output[encodings['input_ids'][j].shape[0]:],
                skip_special_tokens=True
            ).strip()
            
            correct_answer = chr(65 + batch_data[j]['answer'])
            if response.upper().startswith(correct_answer):
                total_correct += 1
            total_samples += 1
    
    return total_correct / total_samples

class BitFlipEnvWithBatch(BitFlipEnv):
    """Environment with batched evaluation"""
    def evaluate_model(self) -> float:
        return batch_mmlu_evaluate(
            self.model,
            self.tokenizer,
            batch_size=self.batch_size
        )