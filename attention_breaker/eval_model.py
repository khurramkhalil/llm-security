import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from typing import List, Dict
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_mmlu_prompt(question: str, choices: List[str]) -> str:
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
            temperature=0.0
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
            prompt = format_mmlu_prompt(
                question['question'],
                question['choices']
            )
            
            # Get model response
            response = get_model_response(model, tokenizer, prompt)
            
            # Check if response matches correct answer
            correct_answer = chr(65 + question['answer'])
            if response.strip().upper().startswith(correct_answer):
                correct += 1
        
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