import warnings
import torch
import os
import json
import argparse
from tqdm.auto import tqdm
from huggingface_hub import login
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness

warnings.simplefilter('ignore')

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            data.append(json.loads(line))
    return data

def read_problems(ds):
    return {task['task_id']: task for task in ds['test']}

def evaluate_samples(
    samples: list,
    problems: dict,
    k: str = "1,10,100",    
    n_workers: int = 12,
    timeout: float = 10.0,
    is_mbpp: bool = False
):  
    return evaluate_functional_correctness(samples, problems, list(map(int, k.split(","))), n_workers, timeout, is_mbpp)

def main(task, model, sample_path):
    print(f'Evaluating model <{model}> on task <{task}>...')
    samples = read_jsonl(sample_path)[0]
    if task == 'humaneval':
        task_id = 'HumanEval'
        problem = 'openai_humaneval'
        samples = [{'task_id': f'HumanEval/{index}', 'completion': string} for index, item in enumerate(samples) for string in item]
    elif task == 'mbpp':
        task_id = 'Mbpp'
        problem = 'mbpp'
        samples = [{'task_id': index+11, 'completion': string} for index, item in enumerate(samples) for string in item]
   
    problems = read_problems(load_dataset(problem))

    results = evaluate_samples(samples, problems, is_mbpp=(task == 'mbpp'))

    results_path = os.path.join(os.path.dirname(sample_path), os.path.basename(sample_path).replace('samples_', 'results_'))
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f'Evaluation results for model <{model}> on task <{task_id}>: *{results}*')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model.')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate (humaneval/mbpp)')
    parser.add_argument('--model', type=str, required=True, help='Path to the sample file')
    parser.add_argument('--sample_path', type=str, required=True, help='Path to the sample file')
    args = parser.parse_args()

    main(args.task, args.model, args.sample_path)
