import torch
import json
import argparse
import random
import os
 
import re
from statistics import mean, stdev

#parse devices type -- cuda, mps
parser = argparse.ArgumentParser(description='Process some data for sleep quality prediction.')
parser.add_argument('--device', type=str, help='Device to run the model on (cuda, mps, cpu)', default='cuda')
parser.add_argument('--max_new_tokens', type=int, default=20)
parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for sampling (0.0 for greedy, >0.0 for sampling)')
parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
parser.add_argument('--num_runs', type=int, default=3, help='Number of runs/seeds (fixed seeds: 0,1,2)')
parser.add_argument('--base_seed', type=int, default=0, help='Base random seed; per-run seed = base_seed + run_idx. Set to -1 to disable seeding.')
parser.add_argument('--use_sampling', type=int, default=0, help='Use sampling (1) or greedy decoding (0)')
# parser.add_argument('--model', type=str, default='phi-3')
parser.add_argument('--pc', type=int, default=1)
parser.add_argument('--lora', type=int, default=1)
parser.add_argument('--int_constraint', type=int, default=0)
parser.add_argument('--max_resamples', type=int, default=3, help='Max resampling attempts to satisfy constraints')
parser.add_argument('--inference_mode', type=str, default='zs', 
                    choices=['zs', 'fs-1', 'fs-3', 'fs-5', 'fs-10'], 
                    help='Inference mode: zs (zero-shot), fs-1 (1 example), fs-3 (3 examples), fs-5 (5 examples), fs-10 (10 examples)')
parser.add_argument('--quantization', type=int, default=1, 
                    help='Enable 4-bit quantization (1) or disable (0) - default: 1')

args = parser.parse_args()

if args.device == 'cuda':
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit()
    print("cuda:", torch.cuda.is_available())
elif args.device == 'mps':
    if not torch.backends.mps.is_available():
        print("MPS is not available. Exiting...")
        exit()
    print("mps:", torch.backends.mps.is_available())

print("inference mode:", args.inference_mode)

# int pc & lora to bool
if args.pc == 1:
    args.pc = True
else:
    args.pc = False
    
if args.lora == 1:
    args.lora = True
else:
    args.lora = False   

if args.use_sampling == 1:
    args.use_sampling = True
else:
    args.use_sampling = False

if args.quantization == 1:
    args.quantization = True
else:
    args.quantization = False

print('pc:', args.pc,'\nlora:', args.lora)
print('max_new_tokens', args.max_new_tokens)
print('temperature:', args.temperature)
print('use_sampling:', args.use_sampling)
print('quantization:', args.quantization)
print('num_runs:', args.num_runs)
print('base_seed:', args.base_seed)


# Free up disk space by clearing the Hugging Face cache
import shutil
from pathlib import Path

def clear_huggingface_cache():
    try:
        # Dynamically get the cache directory
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        
        # Check if the cache directory exists
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cache cleared: {cache_dir}")
        else:
            print(f"No cache found at: {cache_dir}")
    except Exception as e:
        print(f"Error clearing cache: {e}")

# load models & quantization
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Phi3ForCausalLM
from peft import PeftModel

# Set up quantization config based on user choice
if args.quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16
    )
else:
    quantization_config = None

#def extract_model_info(target_model):
    # """
    # Extract the base model name and determine if it is a specific fine-tuned or weighted-loss version.
    # Supports only the following suffixes: _finetuned_1, _finetuned_2, _finetuned_3, _finetuned_4,
    # -weighted-loss_1, -weighted-loss_2, -weighted-loss_3, -weighted-loss_4.
    # Returns None for variant_type and version if it is a base model.
    # """
    # Regular expression to match the base model with optional specific suffix and version
#    match = re.match(
#           r"^(.*?)(?:(_finetuned(?:_\d+)?(?:_default_lora)?|_weighted-loss)(_s?\d+(?:_td)?))?$",
#            target_model
#            )
    
#    if not match:
 #       raise ValueError(f"Invalid model format: {target_model}")
    
#    base = match.group(1)  # Extract the base model name
#    print('*******************************', target_model, '************************************')
#    print('----------------------------------------', base, '--------------------------------------------')
#    is_finetuned = match.group(2)  # "_finetuned", "_fintuned_s or "-weighted-loss", or None for base models
#    version = match.group(3)  # Version number, or None for base models

#    return base, is_finetuned, version

# def get_fine_tuned_model(is_finetuned, base, version, base_model, base_tokenizer, model_name):
#     print('------------------------',base, '-------------------------')
#     base_path = 'Desktop/Finetuned'
#     base_path = "xw17"
#     if is_finetuned == "_finetuned":
#         path = base_path +"/"+base+"_finetuned"
#         if version:
#            path += version
#         tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
#         model = PeftModel.from_pretrained(base_model, path)
       
#     elif is_finetuned == "-weighted-loss":
#         base_path = 'Desktop/Finetuned'
#         path = base_path +"/"+base+"-weighted-loss_" + version
#         tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
#         model = PeftModel.from_pretrained(base_model, path)
       
#     else:
#        tokenizer = base_tokenizer
#        model = base_model
       
#     return tokenizer, model

def get_fine_tuned_model(base_model, base_tokenizer, base, targed_model):
    if base == targed_model:
        tokenizer = base_tokenizer
        model = base_model

    else:
        hub_username = os.environ.get('HUB_USERNAME', 'your_username')
        tokenizer = AutoTokenizer.from_pretrained(f"{hub_username}/{targed_model}", trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, f"{hub_username}/{targed_model}")

    return tokenizer, model

def get_tk_with_model(base, target_model, lora = args.lora):
    if 'Phi-3' in base:
        provider = 'microsoft'
    elif "gemma" in base:
        provider = 'google'
    elif 'SmolLM' in base:    
        provider = 'HuggingFaceTB'
    elif 'Qwen' in base:
        provider = 'Qwen'
    elif 'TinyLlama' in base:
        provider = 'TinyLlama'
    elif 'MiniCPM3' in base:
        provider = 'openbmb'        
    elif 'Llama-3.2' in base:
        provider = 'meta-llama'
    
    # provider = 'Models'
    if lora:
        # base, is_finetuned, version = extract_model_info(target_model)
        base_path = f"{os.environ.get('MODELS_PATH', './models') if args.pc else provider}/{base}"
        # print(base_path)
    # base_path = provider + "/" + base
    else:
        f_path = os.environ.get('HUB_USERNAME', 'your_username') + '/' + target_model
    
    if 'Phi-3-mini-4k-instruct' in target_model or 'Phi-3.5-mini-instruct' in target_model:
        if lora:
            base_model = Phi3ForCausalLM.from_pretrained(base_path, trust_remote_code=True, attn_implementation='eager', torch_dtype="auto", device_map=args.device, quantization_config=quantization_config)
            base_tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        
            # tokenizer, model = get_fine_tuned_model(is_finetuned, base, version, base_model, base_tokenizer, target_model)
            tokenizer, model = get_fine_tuned_model(base_model, base_tokenizer, base, target_model)

        else:
            model = Phi3ForCausalLM.from_pretrained(f_path, trust_remote_code=True, attn_implementation='eager', torch_dtype="auto", device_map=args.device, quantization_config=quantization_config)
            tokenizer = AutoTokenizer.from_pretrained(f_path, trust_remote_code=True)
    
    elif "gemma-2-2b" in target_model:
        # Use environment variable for HF token
        if 'HF_TOKEN' not in os.environ:
            print("Warning: HF_TOKEN environment variable not set. Please set it for Gemma models.")
            print("Example: export HF_TOKEN='your_huggingface_token_here'")
        if lora:
            base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto", device_map=args.device, quantization_config=quantization_config, trust_remote_code=True)
            base_tokenizer = AutoTokenizer.from_pretrained(base_path, use_auth_token=os.environ['HF_TOKEN'], trust_remote_code=True)
            
            # tokenizer, model = get_fine_tuned_model(is_finetuned, base, version, base_model, base_tokenizer, target_model)
            tokenizer, model = get_fine_tuned_model(base_model, base_tokenizer, base, target_model)
            
        else:
            model = AutoModelForCausalLM.from_pretrained(f_path, torch_dtype="auto", device_map=args.device, quantization_config=quantization_config, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(f_path, use_auth_token=os.environ['HF_TOKEN'], trust_remote_code=True)
               
    else:
        if lora:
            base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto", device_map=args.device, quantization_config=quantization_config)
            base_tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

            # tokenizer, model = get_fine_tuned_model(is_finetuned, base, version, base_model, base_tokenizer, target_model)
            tokenizer, model = get_fine_tuned_model(base_model, base_tokenizer, base, target_model)

        else:
            model = AutoModelForCausalLM.from_pretrained(f_path, torch_dtype="auto", device_map=args.device, quantization_config=quantization_config)
            tokenizer = AutoTokenizer.from_pretrained(f_path, trust_remote_code=True)
    return tokenizer, model 

# load data
def json_reader(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data

from datasets import Dataset
def load_json_to_dataset(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            # print(data)
        return Dataset.from_list(data)

def reconstruct_prompt_few_shots(data):
    instruction_few_shot = "You are a health assistant. Your mission is to read the following "+ str(len(data))+" examples and return your prediction based on the health query.\n"
    prompt = instruction_few_shot
    count = 1
    for entry in data:
        instruction = entry.get('instruction')
        # instruction = "You are a health assistant. Your mission is to read the following examples and return your prediction based on the health query"
        input_data = entry.get('question')
        output = entry.get('answer')
        # prompt = f"{instruction}\n{input_data}\n{output}"
        prompt += "\nExample "+str(count)+"\n"
        prompt += f"{instruction} {input_data} {output}\n"
        count+=1
    # prompt += "\nEnd of examples! Now read the following question and return your prediciton based on the knowledge your learned.\n\n"
    prompt += "\nFinally, please answer to the below question:\n"
    return prompt

def reconstruct_prompts_zero_shot(data):
    prompts = []
    for entry in data:
        # if task =='fatigue' or task =='stress' or task =='sleep_quality':
        #     instruction = "Given the user's context and sensor data, predict the "+task+" level(ranges from 1 to 5)"
        # else:
        #     instruction = "Given the user's context and sensor data, predict the readiness score (ranges from 0 to 10)"
        # instruction = "You are a health assistant. Your mission is to read the following input health query and return your prediction."
        instruction = entry.get('instruction')
        input_data = entry.get('question')
        
        # Ensure all numbers are converted to strings in the input
        input_data = re.sub(r'(\d+(\.\d+)?)', r'"\1"', input_data)
        
        ## output = entry.get('output')
        ## prompt = f"{instruction}\n{input_data}\n{output}"
        prompt = f"{instruction} {input_data}"
        
        prompts.append(prompt)
    return prompts

def true_results(data):
    results = []
    for entry in data:
        answer = entry.get('answer')
        result = f"{answer}"
        results.append(result)

    return results

def get_answer_template(data_name: str, task: str) -> str:
    data_name_lower = data_name.lower()
    if data_name_lower == "pmdata":
        return f"The predicted {task} level is "
    if data_name_lower == "aw_fb":
        if task == "activity":
            return "The predicted activity type is "
        if task == "calories":
            return "The predicted calorie burn is "
    if data_name_lower == "globem":
        return f"The predicted PHQ-4 {task} score is "
    return ""

def extract_answer_value(response: str, data_name: str, task: str):
    data_name_lower = data_name.lower()
    # Build regex patterns consistent with model.ipynb
    if data_name_lower == "pmdata":
        pattern = rf"The predicted {re.escape(task)} level is:?\s*(\d+(?:\.\d+)?)[\.]?"
    elif data_name_lower == "aw_fb":
        if task == "activity":
            pattern = r"The predicted activity type is[:\s]*[\[*]*([A-Za-z0-9\s]+)[\]*]*[\.]?"
        else:  # calories
            pattern = r"The predicted calorie burn is (\d+(?:\.\d+)?)[\.]?"
    elif data_name_lower == "globem":
        pattern = rf"The predicted PHQ-4 {re.escape(task)} score is:?\s*(\d+(?:\.\d+)?)[\.]?"
    else:
        # Fallback: any number in the string
        pattern = r"(\d+(?:\.\d+)?)"

    m = re.search(pattern, response)
    if not m:
        # Fallbacks: if template-specific pattern not found
        if data_name_lower == "aw_fb" and task == "activity":
            # Try to capture a reasonable activity label token/phrase
            m2 = re.search(r"([A-Za-z][A-Za-z0-9\s\-]{2,})", response)
            if not m2:
                return None
            value = m2.group(1).strip()
            return value
        else:
            # Numeric fallback: pick the first number anywhere
            m2 = re.search(r"(\d+(?:\.\d+)?)", response)
            if not m2:
                return None
            value = m2.group(1)
            # proceed to cast below
    else:
        value = m.group(1)
    if data_name_lower == "aw_fb" and task == "activity":
        return value.strip()
    # Cast numeric appropriately
    if "." in value:
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return int(value)
    except ValueError:
        return None

def calculate_response_statistics(responses, data_name, task):
    """Calculate mean and standard deviation of numeric responses"""
    numeric_values = []
    for response in responses:
        value = extract_answer_value(response, data_name, task)
        if value is not None:
            if isinstance(value, (int, float)):
                numeric_values.append(value)
    
    if len(numeric_values) > 1:
        return {
            'mean': mean(numeric_values),
            'std': stdev(numeric_values),
            'values': numeric_values,
            'count': len(numeric_values)
        }
    elif len(numeric_values) == 1:
        return {
            'mean': numeric_values[0],
            'std': 0.0,
            'values': numeric_values,
            'count': 1
        }
    else:
        return {
            'mean': None,
            'std': None,
            'values': [],
            'count': 0
        }

def compute_task_metrics(true_values_all, per_run_predictions, metric_type):
    """Compute per-run metric and aggregate mean/std.
    metric_type: 'accuracy' or 'mse'
    """
    import math
    per_run_scores = []
    n_used = 0
    if metric_type == 'accuracy':
        # true_values may be int/float/str; compare by exact match
        n_used = len(true_values_all)
        for preds in per_run_predictions:
            correct = 0
            total = 0
            for y_true, y_pred in zip(true_values_all, preds):
                if y_true is None:
                    continue
                total += 1
                if y_pred is not None and str(y_pred) == str(y_true):
                    correct += 1
            score = (correct / total) if total > 0 else 0.0
            per_run_scores.append(score)
    else:  # mse
        # Only numeric pairs
        for preds in per_run_predictions:
            sqerrs = []
            for y_true, y_pred in zip(true_values_all, preds):
                if isinstance(y_true, (int, float)) and isinstance(y_pred, (int, float)):
                    diff = float(y_pred) - float(y_true)
                    sqerrs.append(diff * diff)
            n_used = len(sqerrs)
            score = (sum(sqerrs) / n_used) if n_used > 0 else float('nan')
            per_run_scores.append(score)

    # Aggregate
    if len(per_run_scores) > 1:
        mean_score = sum(per_run_scores) / len(per_run_scores)
        variance = sum((s - mean_score) ** 2 for s in per_run_scores) / (len(per_run_scores) - 1)
        std_score = math.sqrt(variance)
    elif len(per_run_scores) == 1:
        mean_score = per_run_scores[0]
        std_score = 0.0
    else:
        mean_score = float('nan')
        std_score = float('nan')
    return per_run_scores, mean_score, std_score, n_used

# DATA = ["PMData", "GLOBEM", "AW_FB"] # "PMData", "GLOBEM", "AW_FB"
datasets = ["PMData", "GLOBEM", "AW_FB"] 
# datasets = ["GLOBEM", "AW_FB"] 
# datasets = ["pmdata", "globem", "aw_fb"]
sub_tasks_list = [['fatigue', 'readiness', 'sleep_quality', 'stress'],
                 ["anxiety", "depression"],
                 ["activity", "calories"]]
#sub_tasks = ['fatigue', 'readiness', 'sleep_quality', 'stress']
#sub_tasks = ["stress_resilience", "sleep_disorder"]
#sub_tasks = ["anxiety", "depression"]
#sub_tasks = ["activity", "calories"]

indexs = ['', '', '', '']
# indexs = ['_finetuned_1_default_lora','_finetuned_2_default_lora',
#           '_finetuned_3_default_lora','_finetuned_4_default_lora']
base_models = ['gemma-2-2b-it','Phi-3-mini-4k-instruct', 'SmolLM-1.7B-Instruct','Qwen2-1.5B-Instruct', 
'TinyLlama-1.1B-Chat-v1.0', 'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Phi-3.5-mini-instruct', 'Qwen2.5-1.5B-Instruct']

# models = [f"{model}-weighted-loss_{index}" for model in models]

# models = [f"{model}_finetuned_{index}" for model in models]
# models = ['Llama-3.2-3B-Instruct_finetuned_1']

if args.device == 'cuda':
    if args.pc:
        subtask_dir = "eval"
    else:
        subtask_dir = "health-SLM/eval"
else:
    subtask_dir = "Desktop/health-SLM/eval"


# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
#     ### task:
#     {}

#     ### Instruction:
#     {}

#     ### Input:
#     {}

#     ### Response:
#     {}"""

# def formatting_prompts_func(examples, DATA, task):
#     tasks        = examples['task']
#     instructions = examples["instruction"]
#     inputs       = examples["question"]
#     #outputs      = examples["answer"]
#     texts = []
#     for task, instruction, input in zip(tasks, instructions, inputs):
#         if DATA.lower() == "pmdata":
#            response = "The predicted " + task + " level is "
#         elif DATA.lower() == "aw_fb":
#             if task == 'activity':
#                 response = "The predicted " + task + " type is "
#             elif task == 'calories':
#                 response = "The predicted calorie burn is "
#             if task == 'sleep disorder':
#                 response = "The predicted sleep disorder is "
#             elif task == 'stress resilience':
#                 response = "The predicted stress resilience index is "
#         elif DATA.lower() == "globem":
#             response = "The predicted PHQ-4 " + task + " score is "
#         text = alpaca_prompt.format(task, instruction, input, response)
#         texts.append(text)
#     return { "text" : texts}

# Define the dataset-task mapping and constraints

datasets_tasks_constraints = {
    "PMData": {
        "fatigue": {"type": "int", "constraints": ["0", "1", "2", "3", "4", "5"]},
        "readiness": {"type": "int", "constraints": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]},
        "sleep_quality": {"type": "int", "constraints": ["1", "2", "3", "4", "5"]},
        "stress": {"type": "int", "constraints": ["0", "1", "2", "3", "4", "5"]},
    },
    "GLOBEM": {
        "depression": {"type": "int", "constraints": ["0", "1", "2", "3", "4"]},
        "anxiety": {"type": "int", "constraints": ["0", "1", "2", "3", "4"]},
    },
    "AW_FB": {
        "activity": {"type": "int", "constraints": None},
        "calories": {"type": "float", "constraints": None},  # No constraints
    },
}

# Removed LMQL usage; we will enforce constraints post-hoc via validation and optional resampling


def main(): #start prediction
    for index, sub_tasks, DATA in zip(indexs, sub_tasks_list, datasets):

        models = [f"{model}{index}" for model in base_models]

        for base_model, targed_model in zip(base_models, models):

            #load targed model
            tokenizer, model = get_tk_with_model(base_model, targed_model)
            print("------------------------------------------------------------")
            print("\ncurrent base model:", base_model,'\n')
            print("\ncurrent target model:", targed_model,'\n')

            for task in sub_tasks:
                task_info = datasets_tasks_constraints[DATA].get(task, {})
                constraints = task_info.get("constraints")
                # Skip if destination file already exists
                output_dir = os.path.join('std/zeroshot', f'predictions_{targed_model}')
                out_file = os.path.join(output_dir, f'{DATA.lower()}_{task}.json')
                if os.path.exists(out_file):
                    print(f"Skip: found existing predictions → {out_file}")
                    continue
            
                #load corresponding task data
                sub_path = os.path.join(subtask_dir, DATA.lower()+"_"+task)
                if task == 'activity':
                    sub_path = os.path.join(subtask_dir, DATA.lower()+"_"+task+"_updated")    
                file_path = os.path.join(sub_path, "step1.json")
                eval_data = load_json_to_dataset(file_path)
                #prompts_zero_shots = reconstruct_prompts_zero_shot(eval_data, task)
                
                # prompts_zero_shots = eval_data.map(lambda examples: formatting_prompts_func(examples, DATA, task), batched=True)['text']
                
                prompts_zero_shots = reconstruct_prompts_zero_shot(eval_data)
                results = true_results(eval_data)

                # answer_template = " For example, the answer should be in the following format:\nAnswer: {}."+"\n Your answer:".format(rand_num)
                answer_template = get_answer_template(DATA, task)
                # answer_template = "The answer is"
                # answer_template = " please fill the your predicted "+task+" level in the bracket ()"
                # answer_template_prefix = " Please provide reasons to support your answer!\n\n"
                answer_template_prefix = " "
                # if task == 'readiness':
                #     answer_template_prefix = "(Please give a whole number from 0 to 10)\n"
                # elif task == "stress":
                #     answer_template_prefix = "(Please give a whole number from 0 to 5)\n"
                # elif task == "sleep_quality" or task == "fatigue":
                #     answer_template_prefix = "(Please give a whole number from 1 to 5)\n"
                # elif task == "activity":
                #     answer_template_prefix = "(Choose one of these: 'Self Pace Walk', 'Sitting', 'Lying', 'Running 7 METs', 'Running 5 METs', or 'Running 3 METs')\n"
                # elif task == "calories":
                #     answer_template_prefix = "(Please give a number based on your calculation)\n"
                # elif task == "sleep_disorder":
                #     answer_template_prefix = "(Choose either 1 for 'exists' or 0 for 'does not exist')\n"
                # elif task == "stress_resilience":
                #     answer_template_prefix = "(Please Give a decimal number between 0.2 and 5)\n"
                
                answer_template = answer_template_prefix + answer_template
                zero_shot_outputs = []
                # Collect predictions per run to compute metrics later
                per_run_numeric_preds = []  # for mse-capable tasks
                per_run_label_preds = []    # for accuracy-capable tasks
                
                # start prediciton
                if args.device == 'cuda':
                    prompts_list = prompts_zero_shots
                else:
                    prompts_list = prompts_zero_shots[0:3]
                # Load few-shot examples if inference_mode is few-shot
                few_shot_examples = ""
                if args.inference_mode.startswith('fs-'):
                    print("--------------------few_shot_example_loading--------------------\n")
                    print("--------------------------task-"+task+"----------------------------")
                    # Extract number from fs-X format
                    N = int(args.inference_mode.split('-')[1])
                    train_path = "train/"+DATA.lower()+"_"+task+"_train_"+str(N)+".json"
                    if task == 'activity':
                        train_path = "train/"+DATA.lower()+"_"+task+"_train_"+str(N)+"_updated.json"
                    train_data = json_reader(train_path)
                    few_shot_examples = reconstruct_prompt_few_shots(train_data)
                    print("---------------few_shot_example_loading_successed---------------\n")

                for i, prompt in enumerate(prompts_list):
                    if args.inference_mode.startswith('fs-'):
                        print("--------------------------few_shot-------------------------\n")
                    else:
                        print("--------------------------zero_shot-------------------------\n")
                    if 'Phi-3' in base_model:
                        provider = 'microsoft/'
                    elif "gemma" in base_model:
                        provider = 'google/'
                    elif 'SmolLM' in base_model:    
                        provider = 'HuggingFaceTB/'
                    elif 'Qwen' in base_model:
                        provider = 'Qwen/'
                    elif 'TinyLlama' in base_model:
                        provider = 'TinyLlama/'
                    elif 'MiniCPM3' in base_model:
                        provider = 'openbmb/'        
                    elif 'Llama-3.2' in base_model:
                        provider = 'meta-llama/'
                    
                    # Build the exact input prompt and print it once per item
                    if args.inference_mode.startswith('fs-'):
                        input_prompt = few_shot_examples + prompts_zero_shots[i] + answer_template
                    else:
                        input_prompt = prompts_zero_shots[i] + answer_template
                    print(f"Input:\n{input_prompt}")

                    # Run multiple times and record raw + extracted outputs
                    runs_detailed = []  # store raw + extracted per run
                    for run in range(args.num_runs):
                        # Tokenize input and move to device
                        inputs = tokenizer(input_prompt, return_tensors="pt").to(args.device)

                        # Resample loop to satisfy constraints when requested
                        attempt = 0
                        while True:
                            # Deterministic per-run seeding using fixed seeds 0..2
                            seed = None
                            if args.use_sampling:
                                fixed_seeds = [0, 1, 2]
                                seed = fixed_seeds[run % len(fixed_seeds)]
                                random.seed(seed)
                                torch.manual_seed(seed)
                                if torch.cuda.is_available():
                                    try:
                                        torch.cuda.manual_seed_all(seed)
                                    except Exception:
                                        pass
                            if args.use_sampling:
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    pad_token_id=tokenizer.eos_token_id,
                                )
                            else:
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id,
                                )

                            # Decode only newly generated tokens for cleaner parsing
                            input_length = inputs["input_ids"].shape[-1]
                            generated_ids = outputs[0][input_length:]
                            zero_shot_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                            # Build templated response for consistent parsing with notebook regexes
                            templated_response = get_answer_template(DATA, task) + zero_shot_response

                            # Post-hoc validation against constraints
                            if args.int_constraint and constraints:
                                val = extract_answer_value(templated_response, DATA, task)
                                is_ok = False
                                if val is not None:
                                    # integer tasks use string-list constraints
                                    if isinstance(constraints, dict) and constraints.get('min') is not None:
                                        # continuous range constraint
                                        if isinstance(val, (int, float)):
                                            min_v = float(constraints['min'])
                                            max_v = float(constraints['max'])
                                            is_ok = (min_v <= float(val) <= max_v)
                                    elif isinstance(val, int):
                                        is_ok = str(val) in constraints
                                    else:
                                        # if float with discrete list, check membership else accept
                                        is_ok = (str(val) in constraints) if constraints is not None else True
                                if is_ok or attempt >= args.max_resamples:
                                    break
                                attempt += 1
                                continue
                            else:
                                break

                        # Extract answer per template using templated response
                        extracted_value = extract_answer_value(templated_response, DATA, task)

                        # Store
                        runs_detailed.append({
                            "raw_response": zero_shot_response,
                            "extracted_answer": extracted_value,
                            "seed_used": seed
                        })

                        print(f"\nRun {run+1} - Output:\n<begin>{zero_shot_response}<end>\n\nExtracted: {extracted_value}\n\n")
                    
                    # No statistics/metrics: only record runs

                    # Also keep a flat list of extracted values for this item
                    extracted_values_list = [r.get("extracted_answer") for r in runs_detailed]
                    print(f"Extracted values across runs: {extracted_values_list}")
                    zero_shot_outputs.append({
                        "input_prompt": input_prompt,
                        "runs": runs_detailed,
                        "extracted_values": extracted_values_list,
                        "true_result": results[i]
                    })

                    print("\n************************************************************")
                    #print("--------------------------few_shot-------------------------\n")

                    #prompts_few_shot = prompts_zero_shots[i]+answer_template
                    # print(prompts_few_shot)
                # inputs = tokenizer(few_shot_examples + prompts_few_shot, return_tensors="pt").to(args.device)
                    #outputs = model.generate(**inputs, max_new_tokens = args.max_new_tokens, do_sample=False)
                    #few_shot_response = tokenizer.decode(outputs[0])
                    #print(few_shot_response)

                    #zero_shot_outputs.append({
                    #"few_shot_response":few_shot_response,
                    #"true_result": results[i]
                    #})

                    #print("\n************************************************************")

                    # print("\nfew_shot")
                    # prompts_few_shot = prompts_few_shots[i]+answer_templete
                    # # print(prompts_few_shot)
                    # inputs = tokenizer(prompts_few_shot, return_tensors="pt").to(args.device)
                    # outputs = model.generate(**inputs, max_new_tokens = args.max_new_tokens, temperature = 0.0)
                    # few_shot_response = tokenizer.decode(outputs[0])
                    # print("\nPredicted readiness score:", few_shot_response)
                    # print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    # print("\nThe true results:", results[i])
                    # print("\n------------------------------------------------------------")

                # Compute metrics per task
                # Build true target vector
                true_targets = []
                for t in results:
                    # Try numeric cast first
                    try:
                        if "." in str(t):
                            true_targets.append(float(t))
                        else:
                            true_targets.append(int(t))
                    except Exception:
                        true_targets.append(str(t))

                # Align per-run predictions shape with true targets
                # Flatten collected predictions if needed
                # Note: we built per_run_* within the per-item loop; ensure length matches num_runs
                while len(per_run_numeric_preds) < args.num_runs:
                    per_run_numeric_preds.append([])
                while len(per_run_label_preds) < args.num_runs:
                    per_run_label_preds.append([])

                # export the prediction only (raw + extracted per run)
                if args.inference_mode.startswith('fs-'):
                    N = int(args.inference_mode.split('-')[1])
                    output_dir = os.path.join(f'predictions_few_shot_{N}', f'predictions_{targed_model}')
                else:
                    output_dir = os.path.join('zeroshot', f'predictions_{targed_model}')
                os.makedirs(output_dir, exist_ok = True)
                with open(os.path.join(output_dir, f'{DATA.lower()}_{task}.json'), "w") as outfile_eval:
                    outfile_eval.write(json.dumps(zero_shot_outputs, indent = 4))
                
                # free up disk space
                clear_huggingface_cache()
                
if __name__ == "__main__":
    main()
