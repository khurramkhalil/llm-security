import os
import gc
import csv
import ast
import copy
import logging
from logging_config import setup_logging

import torch
import numpy as np
import pandas as pd
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from attention_breaker.optim_layer_ranking import layer_ranking
from attention_breaker.run_mmlu import main_ as batch_mmlu_evaluate
from attention_breaker.genbfa_optimization import genetic_optimization_top
from attention_breaker.weight_subset_selection import weight_subset_selection

# Set up logging
setup_logging()


# Define the quantization configurTrue
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Set to True for 4-bit quantization
    load_in_8bit=False,  # Set to False for 8-bit quantization
    llm_int8_threshold=6.0,  # Optional: threshold for mixed-precision
    llm_int8_skip_modules=None  # Optional: modules to skip for mixed-precision
)

# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the tokenizer and model from Hugging Face's model hub.
def load_model(model_name):
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer


def main():

    logging.info("Begin execution in main script")

    # Define parameters
    alpha = 0.5
    subsample_rate = 5
    # subsample_rates = [5, 10, 20]
    # top_n_layers = 2
    # loss_threshold = 0.10
    max_generations = 10
    mutation_rate = 0.1
    
    filename = 'critical_weights.csv'

    if not os.path.exists(filename):
        # Load model
        model, tokenizer = load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        original_acc = batch_mmlu_evaluate(model, tokenizer)
        # Extract layer names and numbers
        layer_info = [(name, i, -1) for i, (name, _) in enumerate(model.named_parameters()) if 'weight' in name]

        # Create a DataFrame with the specified columns
        df = pd.DataFrame(layer_info, columns=['layer_name', 'layer_number', 'accuracy'])

        # Add columns for critical weights, initialized to -1
        for i in range(1, 6):
            df[f'critical_weight_{i}'] = -1

        # Create a new row to add at the top
        new_row = pd.DataFrame([{
            'layer_name': 'all.layers',
            'layer_number': -1,
            'accuracy': original_acc,
            'critical_weight_1': 0,
            'critical_weight_2': 1,
            'critical_weight_3': 2,
            'critical_weight_4': 3,
            'critical_weight_5': 4
        }])

        # Concatenate the new row with the existing DataFrame
        df = pd.concat([new_row, df], ignore_index=True)

        # Save the DataFrame to 'critical_weights.csv'
        df.to_csv('critical_weights.csv', index=False)
        

    else:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)        

    # Shuffle the DataFrame
    shuffled_df = df.sample(frac=1, random_state=np.random.randint(0, 10000))

    # Iterate over each row in the shuffled DataFrame
    for index, row in shuffled_df.iterrows():

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename) 

        # Log results
        logging.info(f"Evaluating through the row: {row['layer_number']}, with name: {row['layer_name']}")

        if df.iloc[index]['critical_weight_1'] == -1:
            # Load fresh copy of the model
            model, tokenizer = load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            print(f"Index: {index}")

            sensitivity_losses = layer_ranking(model, tokenizer, alpha, subsample_rate, row['layer_name'])
            print("Layer ranking completed. Sensitive layers identified.")

            # Explicitly delete the model and clear cache
            del model
            gc.collect()  # Call garbage collector
            torch.cuda.empty_cache()

            # Load fresh copy of the model
            model, tokenizer = load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            # Step 3: Genetic Optimization
            print("Optimizing weight subset...")

            best_solution, top_indices, top_acc = genetic_optimization_top(model, sensitivity_losses[0], tokenizer, max_generations, mutation_rate)
            print(f"Optimized weight subset: {best_solution}")
            
            logging.info(f"Optimized weight subset: {best_solution}")

            df.loc[df['layer_number'] == row['layer_number'], 'accuracy'] = top_acc
            df.loc[df['layer_number'] == row['layer_number'], 'critical_weight_1'] = top_indices[0]
            df.loc[df['layer_number'] == row['layer_number'], 'critical_weight_2'] = top_indices[1]
            df.loc[df['layer_number'] == row['layer_number'], 'critical_weight_3'] = top_indices[2]
            df.loc[df['layer_number'] == row['layer_number'], 'critical_weight_4'] = top_indices[3]
            df.loc[df['layer_number'] == row['layer_number'], 'critical_weight_5'] = top_indices[4]

            df.to_csv('critical_weights.csv', index=False)
            
            print(f"Optimized weight subset: {best_solution}")

if __name__ == "__main__":
    main()