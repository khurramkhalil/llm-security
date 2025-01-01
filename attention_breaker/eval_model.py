import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from typing import List, Dict
import torch.nn.functional as F

import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to format the MMLU dataset prompt
def format_mmlu_prompt(data):
    # Extract question, subject, choices, and answer from the data
    question = data['question']
    subject = data['subject']
    choices = data['choices']
    answer_idx = data['answer']
    
    # Create a prompt that asks the model to provide only the correct answer
    prompt = f"You are a helpful AI assistant that answer the questions. \nQuestion: {question}\nChoices:\n"
    
    # Add the choices to the prompt
    for i, choice in enumerate(choices):
        prompt += f"{i + 1}. {choice}\n"
    
    # Add instruction to give only the correct answer
    prompt += "\nPlease provide only the correct answer from the choices"

    # Return the formatted prompt and the correct answer
    return prompt, str(answer_idx + 1)  # answer_idx is 0-based, so adding 1


def format_mmlu_prompt_old(question: str, choices: List[str]) -> str:
    """Format the question and choices into a prompt."""
    formatted_choices = "\n".join([f"{chr(65 + i)}. {choice}" 
                                 for i, choice in enumerate(choices)])
    prompt = f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"
    return prompt
# def format_mmlu_prompt(question: str, choices: List[str]) -> str:
#     """Format the question and choices into a prompt."""
#     formatted_choices = "\n".join([f"{chr(65 + i)}. {choice}" 
#                                    for i, choice in enumerate(choices)])
#     prompt = f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"
#     return prompt
def get_model_response(model, tokenizer, prompt: str, max_new_tokens: int = 5) -> str:
    """Get model's response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.01,
            # do_sample=False
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                              skip_special_tokens=True).strip()
    return response

def evaluate_mmlu(model, tokenizer, subjects: List[str] = None, num_samples: int = None):
    """Evaluate model on MMLU benchmark."""
    # Load MMLU validation dataset
    dataset = load_dataset("cais/mmlu", "all")["validation"]
    
    if subjects:
        dataset = dataset.filter(lambda x: x['subject'] in subjects)
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    results = {
        'total_correct': 0,
        'total_questions': 0,
        'subject_scores': {}
    }
    
    # Group questions by subject
    subject_questions = {}
    for item in dataset:
        subject = item['subject']
        if subject not in subject_questions:
            subject_questions[subject] = []
        subject_questions[subject].append(item)
    
    # Evaluate each subject
    for subject, questions in tqdm(subject_questions.items(), desc="Evaluating subjects"):
        correct = 0
        total = len(questions)
        
        for question in tqdm(questions, desc=f"Processing {subject}", leave=False):
            # Format prompt
            prompt, correct_answer = format_mmlu_prompt(question)
            print("Formatted prompt", prompt)
            # Get model response
            response = get_model_response(model, tokenizer, prompt)
            print(f"Response: {response}")
            # pdb.set_trace()
            
            # Compare the predicted answer with the correct answer
            if question['choices'][question['answer']] in response:
                correct += 1

            # if response == correct_answer:
            #     correct += 1
            
            # # Check if response matches correct answer
            # correct_answer = chr(65 + question['answer'])
            # print(f"Correct Answer:  {correct_answer}     ######, Generated Answer: {response.strip().upper()}\n")
            # if response.strip().upper().startswith(correct_answer):
            #     correct += 1
        
        # Calculate and store subject score
        subject_score = (correct / total) * 100
        results['subject_scores'][subject] = {
            'correct': correct,
            'total': total,
            'accuracy': subject_score
        }
        
        results['total_correct'] += correct
        results['total_questions'] += total
    
    # Calculate overall accuracy
    results['overall_accuracy'] = (results['total_correct'] / results['total_questions']) * 100
    
    return results


def batch_mmlu_evaluate(model, tokenizer, batch_size=64, num_samples=128):
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
        prompts = [format_mmlu_prompt(item[0], item[1]) for item in batch_data.items()]
        
        # Tokenize all prompts
        encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            padding_side='left'
        ).to(model.device)
        
        # Generate answers in batch
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id,
                # temperature=0.0,
                do_sample=False
            )
        
        # Process responses
        for jj, output in enumerate(outputs):
            # pdb.set_trace()        
            response = tokenizer.decode(
                output[encodings['input_ids'][jj].shape[0]:],
                skip_special_tokens=True
            ).strip()
            
            # pdb.set_trace()        
            
            correct_answer = chr(65 + batch_data['answer'][jj])
            if response.upper().startswith(correct_answer):
                total_correct += 1
            total_samples += 1
    
    print("Current Accuracy: ", total_correct / total_samples)

    return total_correct / total_samples


def mmlu_evaluate(model, tokenizer):
    # Run evaluation
    # print("Starting MMLU evaluation...")
    results = evaluate_mmlu(
        model, 
        tokenizer,
        num_samples=100  # Limit samples for testing, remove for full evaluation
    )

    # # Print results
    # print("\nOverall Results:")
    # print(f"Total Accuracy: {results['overall_accuracy']:.2f}%")
    # print(f"Total Correct: {results['total_correct']}/{results['total_questions']}")

    return results['overall_accuracy'] / 100


def mmlu_evaluate_extended(model, tokenizer):
    # Run evaluation
    print("Starting MMLU evaluation...")
    results = evaluate_mmlu(
        model, 
        tokenizer,
        num_samples=100  # Limit samples for testing, remove for full evaluation
    )

    # Print results
    print("\nOverall Results:")
    print(f"Total Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Total Correct: {results['total_correct']}/{results['total_questions']}")

    print("\nResults by Subject:")
    subject_df = pd.DataFrame([
        {
            'Subject': subject,
            'Accuracy': data['accuracy'],
            'Correct': data['correct'],
            'Total': data['total']
        }
        for subject, data in results['subject_scores'].items()
    ])

    # Sort by accuracy descending
    subject_df = subject_df.sort_values('Accuracy', ascending=False)
    print(subject_df.to_string(index=False))

    return results


# # Run evaluation
# print("Starting MMLU evaluation...")
# results = evaluate_mmlu(
#     model, 
#     tokenizer,
#     num_samples=100  # Limit samples for testing, remove for full evaluation
# )

# # Print results
# print("\nOverall Results:")
# print(f"Total Accuracy: {results['overall_accuracy']:.2f}%")
# print(f"Total Correct: {results['total_correct']}/{results['total_questions']}")

# print("\nResults by Subject:")
# subject_df = pd.DataFrame([
#     {
#         'Subject': subject,
#         'Accuracy': data['accuracy'],
#         'Correct': data['correct'],
#         'Total': data['total']
#     }
#     for subject, data in results['subject_scores'].items()
# ])

# # Sort by accuracy descending
# subject_df = subject_df.sort_values('Accuracy', ascending=False)
# print(subject_df.to_string(index=False))