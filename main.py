import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from generation import generate_without_remind, generate_with_remind, MAX_THINK_TOKENS, MAX_ANSWER_TOKENS
from data_loader import load_data, get_query
from evaluator import evaluate_responses
from REMIND_PHRASE import REMIND, ADAPTIVE_SYSTEM_BEGIN, ADAPTIVE_SYSTEM_END

def main():
    parser = argparse.ArgumentParser()
    
    # Argument
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--dataset-name", choices=["JBB", "JBB_benign", "HarmBench", "AdvBench", "XSTest", "XSTest_benign", "MATH", "GPQA"], required=True, type=str)
    parser.add_argument("--remind", choices=["none", "entropy", "system_prompt", "begin_prompt", "end_prompt", "begin_think", "end_think", "begin_answer"], required=True, type=str)
    parser.add_argument("--criteria", choices=["gt", "lt"], default="lt", type=str)
    parser.add_argument("--threshold", default=0.5 , type=float)
    parser.add_argument("--max-num-remind", default=1, type=int)
    parser.add_argument("--adaptive", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--evaluator", choices=["RR", "LG3", "LG4"], default="RR", type=str)
    
    args = parser.parse_args()
    
    model_name = args.model_name
    dataset_name = args.dataset_name
    remind = args.remind
    threshold = args.threshold
    criteria = args.criteria
    max_num_remind = args.max_num_remind
    adaptive = args.adaptive
    evaluator = args.evaluator
    
    if args.evaluate:
        # Load generation result
        if remind == "none":
            file_name = f"./response/{dataset_name}/{model_name.split('/')[1]}/{remind}.jsonl"
        else:
            file_name = f"./response/{dataset_name}/{model_name.split('/')[1]}/{remind}_{criteria}_{threshold}_{max_num_remind}_{adaptive}.jsonl"
        with open(file_name, 'r') as f:
            responses = [json.loads(i) for i in f.readlines()]
        
        # Evaluating responses
        evaluated_responses = evaluate_responses(responses, evaluator)
        
        # Preparing to save the evaluated output
        os.makedirs(f"./evaluated/{dataset_name}/{model_name.split('/')[1]}/{evaluator}", exist_ok=True)
        if remind == "none":
            output_file_name = f"./evaluated/{dataset_name}/{model_name.split('/')[1]}/{evaluator}/{remind}.jsonl"
        else:
            output_file_name = f"./evaluated/{dataset_name}/{model_name.split('/')[1]}/{evaluator}/{remind}_{criteria}_{threshold}_{max_num_remind}_{adaptive}.jsonl"
        
        # Saving evaluated results
        with open(output_file_name, "w") as f:
            for response in evaluated_responses:
                json.dump(response, f)
                f.write("\n")
    else:
        # Loading model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )
        model.eval()
        
        # Loading dataset
        dataset = load_data(dataset_name)
        
        # Preparing to save the generated output
        os.makedirs(f"./response/{dataset_name}/{model_name.split('/')[1]}", exist_ok=True)
        if remind == "none":
            file_name = f"./response/{dataset_name}/{model_name.split('/')[1]}/{remind}.jsonl"
        else:
            file_name = f"./response/{dataset_name}/{model_name.split('/')[1]}/{remind}_{criteria}_{threshold}_{max_num_remind}_{adaptive}.jsonl"

        safety_remind = REMIND
        with open(file_name, 'w') as f:
            for idx, data in enumerate(tqdm(dataset)):
                # Getting query
                query = get_query(data, dataset_name)
                
                # Adaptive reminding phrases generation
                if adaptive:
                    initial_response = generate_without_remind(model, tokenizer, ADAPTIVE_SYSTEM_BEGIN+query+ADAPTIVE_SYSTEM_END, MAX_THINK_TOKENS, MAX_ANSWER_TOKENS)["answer"]
                    safety_remind = initial_response.split("<｜end▁of▁sentence｜>")[0].split("\n")[-1]
                    
                # Generating responses
                if remind == "none":
                    result = generate_without_remind(
                            model,
                            tokenizer,
                            query,
                            MAX_THINK_TOKENS,
                            MAX_ANSWER_TOKENS
                        )
                else:
                    result = generate_with_remind(
                            model,
                            tokenizer,
                            query,
                            MAX_THINK_TOKENS,
                            MAX_ANSWER_TOKENS,
                            safety_remind,
                            remind,
                            criteria,
                            threshold,
                            max_num_remind,
                            adaptive
                        )

                # Recording answers for GPQA
                if dataset_name == "GPQA":
                    result["correct_index"] = data.correct_index
                
                # Saving output
                json.dump(result, f)
                f.write("\n")

if __name__ == "__main__":
    main()
