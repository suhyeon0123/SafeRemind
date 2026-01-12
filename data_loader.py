"""
Data loading utilities for SafeRemind.
Handles loading and preprocessing of various datasets.
"""

from datasets import load_dataset
from datetime import datetime
from collections import namedtuple
import random


def load_data(dataset_name):
    """
    Load dataset based on dataset name.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Dataset object
    """
    # Loading dataset
    if dataset_name == "JBB":
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
    elif dataset_name == "JBB_benign":
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["benign"]
    elif dataset_name == "HarmBench":
        dataset = load_dataset("walledai/HarmBench", "standard", split="train")
    elif dataset_name == "AdvBench":
        dataset = load_dataset("walledai/AdvBench", split="train")
    elif dataset_name == "XSTest": # 200
        dataset = load_dataset("walledai/XSTest", split="test")
        dataset = dataset.filter(lambda x: x["label"]=="unsafe")
    elif dataset_name == "XSTest_benign": # 250
        dataset = load_dataset("walledai/XSTest", split="test")
        dataset = dataset.filter(lambda x: x["label"]=="safe")
    elif dataset_name == "AIME":
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    elif dataset_name == "MATH":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif dataset_name == "GPQA":
        raw_dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond', split="train")
        Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])
        random.seed(43)
        dataset = []
        for row in raw_dataset:
            list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
            random.shuffle(list_choices)
            example = Example(row["Question"], list_choices[0], list_choices[1], list_choices[2], list_choices[3], list_choices.index(row['Correct Answer']))
            dataset.append(example)
    elif dataset_name == "LiveCodeBench":
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5", split="test", trust_remote_code=True)
        dataset = dataset.filter(lambda x: datetime(2024,7,31)<datetime(*[int(j) for j in x["contest_date"].split("T")[0].split("-")])<datetime(2025,2,1))
    else:
        print(f"Dataset {dataset_name} is not implemented yet.")
        exit(1)
    return dataset


def get_query(data, dataset_name):
    """
    Extract query from dataset entry.
    
    Args:
        data: Dataset entry
        dataset_name: Name of the dataset
        
    Returns:
        Query string
    """
    # Getting query
    if "JBB" in dataset_name:
        query = data['Goal']
    elif dataset_name == "HarmBench":
        query = data["prompt"]
    elif dataset_name == "AdvBench":
        query = data["prompt"]
    elif "XSTest" in dataset_name:
        query = data["prompt"]
    elif dataset_name == "MATH":
        query = data["problem"]
    elif dataset_name == "GPQA":
        query = f"What is the correct answer to this question: {data.question}"
        query += f"\n\nChoices:\n(A) {data.choice1}\n(B) {data.choice2}\n(C) {data.choice3}\n(D) {data.choice4}"
    else:
        print(f"Dataset {dataset_name} is not implemented yet.")
        exit(1)
    return query
