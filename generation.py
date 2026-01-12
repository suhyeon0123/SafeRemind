"""
Generation functions for SafeRemind.
Handles both generation with and without safety reminders.
"""

import torch
from transformers import AutoTokenizer


MAX_THINK_TOKENS = 4096
MAX_ANSWER_TOKENS = 1024


def generate_without_remind(
        model,
        tokenizer,
        query,
        MAX_THINK_TOKENS, 
        MAX_ANSWER_TOKENS
    ):
    """
    Generate response without safety reminders.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        query: Input query string
        MAX_THINK_TOKENS: Maximum tokens for thinking step
        MAX_ANSWER_TOKENS: Maximum tokens for answer
        
    Returns:
        dict: Contains query, input, thinking_step, and answer
    """
    # Tokenizing input
    input_template = tokenizer.apply_chat_template([{"role":"user","content":query}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_template, add_special_tokens=False, return_tensors="pt").to('cuda')
    input_ids = inputs["input_ids"]

    # Thinking step generation with greedy decoding
    response_think = ""
    past_key_values = None
    step = 0
    while step < MAX_THINK_TOKENS:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        # Selecting the next token
        logits  = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_id.item()
        response_think += tokenizer.decode(next_id[0])
        step += 1
        
        # Checking stop condition
        if token_id == tokenizer.eos_token_id or "</think>" in response_think:
            break
        
        # Preparing for the next step
        past_key_values = outputs.past_key_values
        input_ids = next_id

    # Ensuring that </think> is at the end of thinking
    thinking_step = response_think.split("</think>")[0]
    response_think = thinking_step+"</think>"
    
    # Tokenizing the input to generate answers
    inputs = tokenizer(input_template+response_think, add_special_tokens=False, return_tensors="pt").to('cuda')
    input_ids = inputs["input_ids"]
    
    # Answer generation with greedy decoding
    response_answer = ""
    past_key_values = None
    step = 0
    while step < MAX_ANSWER_TOKENS:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        # Selecting the next token
        logits  = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_id.item()
        response_answer += tokenizer.decode(next_id[0])
        step += 1
        
        # Checking stop condition
        if token_id == tokenizer.eos_token_id:
            break
        
        # Preparing for the next step
        input_ids = next_id
        past_key_values = outputs.past_key_values

    # Return result
    result = {
        'query':query,
        'input':input_template,
        'thinking_step':thinking_step,
        'answer':response_answer
    }
    return result


def generate_with_remind(
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
    ):
    """
    Generate response with safety reminders.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        query: Input query string
        MAX_THINK_TOKENS: Maximum tokens for thinking step
        MAX_ANSWER_TOKENS: Maximum tokens for answer
        safety_remind: Safety reminder phrase
        remind: Reminder injection strategy
        criteria: Entropy criteria ("gt" or "lt")
        threshold: Entropy threshold value
        max_num_remind: Maximum number of reminders
        adaptive: Whether to use adaptive reminding
        
    Returns:
        dict: Contains query, input, thinking_step, answer, and trigger_count
    """
    # Tokenizing input
    if remind == "system_prompt":
        input_template = tokenizer.apply_chat_template([{"role":"system","content":safety_remind}, {"role":"user","content":query}], tokenize=False, add_generation_prompt=True)
    elif remind == "begin_prompt":
        input_template = tokenizer.apply_chat_template([{"role":"user","content":safety_remind+query}], tokenize=False, add_generation_prompt=True)
    elif remind == "end_prompt":
        input_template = tokenizer.apply_chat_template([{"role":"user","content":query+safety_remind}], tokenize=False, add_generation_prompt=True)
    elif remind == "begin_think":
        input_template = tokenizer.apply_chat_template([{"role":"user","content":query}], tokenize=False, add_generation_prompt=True)+safety_remind
    else:
        input_template = tokenizer.apply_chat_template([{"role":"user","content":query}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_template, add_special_tokens=False, return_tensors="pt").to('cuda')
    input_ids = inputs["input_ids"]

    # Thinking step generation with greedy decoding
    response_think = ""
    past_key_values = None
    step = 0
    if remind == "entropy":
        trigger_count = 0
    else:
        trigger_count = 1
    while step < MAX_THINK_TOKENS:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Selecting the next token
        logits  = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_id.item()
        response_think += tokenizer.decode(next_id[0])
        step += 1
        
        # Checking stop condition
        if token_id == tokenizer.eos_token_id or "</think>" in response_think:
            break
        
        # Preparing for the next step
        past_key_values = outputs.past_key_values
        input_ids = next_id
        
        if remind == "entropy" and response_think.endswith("\n") and trigger_count < max_num_remind:
            # Calculating mean entropy
            prob = torch.softmax(logits, dim=-1)
            entropy = torch.log(prob)*prob
            mean_entropy = torch.mean(-torch.sum(torch.where(torch.isnan(entropy), torch.tensor(0.0), entropy), axis=-1))
            
            # Checking whether the injection criteria is met
            if (criteria=="gt" and mean_entropy > threshold) or (criteria=="lt" and mean_entropy < threshold):
                trigger_count += 1
                response_think += safety_remind
                inputs = tokenizer(input_template+response_think, return_tensors="pt", add_special_tokens=False).to('cuda')
                input_ids = inputs["input_ids"]
                past_key_values=None
                torch.cuda.empty_cache()
            
    # Ensuring that </think> is at the end of thinking
    thinking_step = response_think.split("</think>")[0]
    
    response_answer = ""
    if remind == "end_think":        
        thinking_step += safety_remind
        response_think = thinking_step+"</think>"
    elif remind == "begin_answer":
        response_think = thinking_step+"</think>"+safety_remind
        response_answer = safety_remind
    else:
        response_think = thinking_step+"</think>"

    # Tokenizing the input to generate answers
    inputs = tokenizer(input_template+response_think, add_special_tokens=False, return_tensors="pt").to('cuda')
    input_ids = inputs["input_ids"]
    
    # Answer generation with greedy decoding
    past_key_values = None
    step = 0
    while step < MAX_ANSWER_TOKENS:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        # Selecting the next token
        logits  = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_id.item()
        response_answer += tokenizer.decode(next_id[0])
        step += 1
        
        # Checking stop condition
        if token_id == tokenizer.eos_token_id:
            break
        
        # Preparing for the next step
        input_ids = next_id
        past_key_values = outputs.past_key_values

    # Return result
    result = {
        'query':query,
        'input':input_template,
        'thinking_step':thinking_step,
        'answer':response_answer,
        'trigger_count':trigger_count
    }
    return result
