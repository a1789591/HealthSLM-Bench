# from unsloth import FastLanguageModel
from sklearn.model_selection import train_test_split
import torch
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import DatasetDict

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
num_train_epochs = 3
learning_rate = 5e-5
lora = True
models = ['gemma-2-2b-it', 'Phi-3-mini-4k-instruct', 'SmolLM-1.7B-Instruct','Qwen2-1.5B-Instruct', 
'TinyLlama-1.1B-Chat-v1.0', 'Llama-3.2-1B-Instruct', 
'Llama-3.2-3B-Instruct', 'Phi-3.5-mini-instruct', 'Qwen2.5-1.5B-Instruct']

# models = ['TinyLlama-1.1B-Chat-v1.0']
datasets = ['pmdata', 'globem', 'aw_fb']
index = ['1', '2', '3']
#index = ['']
#datasets = ['universal']
metrics_dir = "metrics_optimized1_task_grouping_off"

if lora:
    index = [item + '_optimized1_task_grouping_off_lora' for item in index]
    metrics_dir += '_lora'
    #learning_rate = 2e-4

# os.chdir("/path/to/your/project/")  # Uncomment and set your project path
os.makedirs(metrics_dir, exist_ok=True)

for m in models:

    if 'TinyLlama' in m:
        path = "TinyLlama/"+m
    elif 'gemma' in m:
        path = "google/"+m
    elif 'Phi-3' in m:
        path = "microsoft/"+m
    elif 'Qwen' in m:
        path = "Qwen/"+m
    elif 'Llama-3.2' in m:
        path = "meta-llama/"+m
    elif 'SmolLM' in m:
        path = "HuggingFaceTB/"+m

    path = "Models/"+m

    import json
    from datasets import Dataset, concatenate_datasets

    # Function to load and convert JSON to Hugging Face dataset
    def load_json_to_dataset(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            # print(data)
        return Dataset.from_list(data)

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
 
    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    # PMData load  
    fatigue_dataset = load_json_to_dataset("health-SLM/train/pmdata_fatigue_train_all_augmentation.json")                   
    # fatigue_train, fatigue_eval = fatigue_dataset.train_test_split(test_size = 0.10, seed = 42).values() 
                                                                                                                                             
    readiness_dataset = load_json_to_dataset("health-SLM/train/pmdata_readiness_train_all_augmentation.json")               
    # #readiness_train, readiness_eval = readiness_dataset.train_test_split(test_size = 0.10, seed = 42).values()        
                                                                                                                                 
    sleep_quality_dataset = load_json_to_dataset("health-SLM/train/pmdata_sleep_quality_train_all_augmentation.json")       
    #sleep_quality_train, sleep_quality_eval = sleep_quality_dataset.train_test_split(test_size = 0.10, seed = 42).values()

    stress_dataset = load_json_to_dataset("health-SLM/train/pmdata_stress_train_all_augmentation.json")
    #stress_train, stress_eval = stress_dataset.train_test_split(test_size = 0.10, seed = 42).values()
    
    # GLOBEM load
    anxiety_dataset = load_json_to_dataset("health-SLM/train/globem_anxiety_train_all_augmentation.json")
    #anxiety_train, anxiety_eval = anxiety_dataset.train_test_split(test_size = 0.10, seed = 42).values()

    depression_dataset = load_json_to_dataset("health-SLM/train/globem_depression_train_all_augmentation.json")
    #depression_train, depression_eval = depression_dataset.train_test_split(test_size = 0.10, seed = 42).values()

    # AW_FB load
    activity_dataset = load_json_to_dataset("health-SLM/train/aw_fb_activity_train_all_updated.json")
    #activity_train, activity_eval = activity_dataset.train_test_split(test_size = 0.10, seed = 42).values()

    calories_dataset = load_json_to_dataset("health-SLM/train/aw_fb_calories_train_all.json")
    #calories_train, calories_eval = calories_dataset.train_test_split(test_size = 0.10, seed = 42).values()

    for n, data in enumerate(datasets):
      if data == 'pmdata':
        #train_dataset = concatenate_datasets([fatigue_train, readiness_train, sleep_quality_train, stress_train])
        #eval_dataset = concatenate_datasets([fatigue_eval, readiness_eval, sleep_quality_eval, stress_eval])
        pmdata_dataset = concatenate_datasets([fatigue_dataset, readiness_dataset, sleep_quality_dataset, stress_dataset])
        train_dataset, eval_dataset = pmdata_dataset.train_test_split(test_size = 0.10, seed = 42).values()
        num_train_epochs = 3
      elif data == 'globem':
        #train_dataset = concatenate_datasets([anxiety_train, depression_train])
        #eval_dataset = concatenate_datasets([anxiety_eval, depression_eval])
        globem_dataset = concatenate_datasets([anxiety_dataset, depression_dataset])
        train_dataset, eval_dataset = globem_dataset.train_test_split(test_size = 0.10, seed = 42).values()
        num_train_epochs = 3
      elif data == 'aw_fb':
        #train_dataset = concatenate_datasets([activity_train, calories_train])
        #eval_dataset = concatenate_datasets([activity_eval, calories_eval])
        aw_fb_dataset = concatenate_datasets([activity_dataset, calories_dataset])
        train_dataset, eval_dataset = aw_fb_dataset.train_test_split(test_size = 0.10, seed = 42).values()
        num_train_epochs = 3
    #   elif data == 'universal':
    #       train_dataset = concatenate_datasets([fatigue_train, readiness_train, sleep_quality_train, stress_train, stress_resilience_train, sleep_disorder_train, stress_resilience_train, sleep_disorder_train, anxiety_train, depression_train, activity_train, calories_train])
    #       eval_dataset = concatenate_datasets([fatigue_eval, readiness_eval, sleep_quality_eval, stress_eval, stress_resilience_eval, sleep_disorder_eval, anxiety_eval, depression_eval, activity_eval, calories_eval])
    #       num_train_epochs = 3

      # combined_dataset = concatenate_datasets([fatigue_dataset, readiness_dataset, sleep_quality_dataset, stress_dataset,\
      #                 stress_resilience_dataset, sleep_disorder_dataset, activity_dataset, calories_dataset])
      
      # Use environment variable for HF token
      hf_token = os.environ.get('HF_TOKEN')
      if not hf_token:
          print("Warning: HF_TOKEN environment variable not set. Please set it for model loading.")
          print("Example: export HF_TOKEN='your_huggingface_token_here'")
          hf_token = None
      
      model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", torch_dtype="auto",
                                                  trust_remote_code=True,
                                                  token=hf_token)
      tokenizer = AutoTokenizer.from_pretrained(path, token=hf_token)

      if tokenizer.pad_token is None:
          tokenizer.pad_token = tokenizer.eos_token

      from transformers import AutoModelForCausalLM, AutoTokenizer
      from peft import LoraConfig, get_peft_model
      #target_modules = ["q_proj", "v_proj"]
      #if "Phi-3" in m:
      #   target_modules = ['qkv_proj']
      target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
      lora_config = LoraConfig(r = 8, lora_alpha = 16,
            target_modules = target_modules,
            lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
         )
      model = get_peft_model(model, lora_config)

      EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
  #    def formatting_prompts_func(examples):
  #        tasks        = examples['task']
  #        instructions = examples["instruction"]
  #        inputs       = examples["question"]
  #        outputs      = examples["answer"]
  #        texts = []
  #        for task, instruction, input, output in zip(tasks, instructions, inputs, outputs):
              # Must add EOS_TOKEN, otherwise your generation will go on forever!
  #            text = alpaca_prompt.format(task, instruction, input, output) + EOS_TOKEN
  #            texts.append(text)
  #        return { "text" : texts, }
  #    pass
      def formatting_prompts_func(examples):
          instructions = examples["instruction"]
          inputs       = examples["question"]
          outputs      = examples["answer"]
          texts = []
          for instruction, input, output in zip(instructions, inputs, outputs):
              # Must add EOS_TOKEN, otherwise your generation will go on forever!
              text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
              #text = instruction + " "+ input+ " " + output + EOS_TOKEN
              texts.append(text)
          return { "text" : texts, }
      pass

      train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
      eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
        
      # Prepare metrics file for this model and dataset
      metrics_file = os.path.join(metrics_dir, m+"_"+data+"_training_metrics.json")

      from trl import SFTTrainer
      from transformers import TrainingArguments
      # from unsloth import is_bfloat16_supported

      import torch
      gpu_stats = torch.cuda.get_device_properties(0)
      start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
      max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
      print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
      print(f"{start_gpu_memory} GB of memory reserved.")

      # split_datasets = dataset.train_test_split(test_size=0.10, seed=42)  # Split into 90% train, 10% test

      # # Access the train and eval datasets
      #train_dataset = split_datasets["train"]
      #eval_dataset = split_datasets["test"]
     
      from math import floor
     # Calculate the total number of training steps
      total_training_steps = floor(len(train_dataset) / (4 * 2)) * num_train_epochs  # Batch size * gradient accumulation * epochs

     # Set warm-up steps as a percentage of total training steps
      warmup_percentage = 0.05  # 5% of the training steps
      eval_percentage = 0.05 # 5% of the training steps 
      dynamic_warmup_steps = int(total_training_steps * warmup_percentage)
      dynamic_eval_steps = int(total_training_steps * eval_percentage)
          
      trainer = SFTTrainer(
          model = model,
          tokenizer = tokenizer,
          train_dataset = train_dataset,
          eval_dataset= eval_dataset,
          dataset_text_field = "text",
          max_seq_length = max_seq_length,
          dataset_num_proc = 2,
          packing = False, # Can make training 5x faster for short sequences
          args = TrainingArguments(
              per_device_train_batch_size = 2,
              gradient_accumulation_steps = 4,
              warmup_steps = dynamic_warmup_steps,
              num_train_epochs = num_train_epochs, # Set this for 1 full training run
              eval_steps = dynamic_eval_steps, # Evaluate every 5% training steps
              save_steps = dynamic_eval_steps, # save after each evaluation
              learning_rate = learning_rate,
              fp16 = False,
              bf16 = True,
              logging_steps = 1,
              optim = "adamw_torch",
              weight_decay = 0.01,
              lr_scheduler_type = "cosine", # linear, cosine
              seed = 3407,
              output_dir = "outputs",
              eval_strategy="steps",
              #load_best_model_at_end=True,  # Load the best model after training stops
              save_total_limit=2,  # Limit checkpoint saves to save disk space
          ),
          #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop after 3 no-improvement evaluations
      )
      trainer_stats = trainer.train()

      used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
      used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
      used_percentage = round(used_memory/max_memory*100, 3)
      lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
      print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
      print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
      print(f"Peak reserved memory = {used_memory} GB.")
      print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
      print(f"Peak reserved memory % of max memory = {used_percentage} %.")
      print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

      # # os.chdir("/mnt/c/Users/Xin Wang/Desktop")
     
      log_history = trainer.state.log_history

      training_loss_history = [entry for entry in log_history if 'loss' in entry]
      training_steps = [entry["step"] for entry in training_loss_history]
      training_losses = [entry["loss"] for entry in training_loss_history]

      evaluation_loss_history = [entry for entry in log_history if 'eval_loss' in entry]
      evaluation_steps = [entry["step"] for entry in evaluation_loss_history]
      evaluation_losses = [entry["eval_loss"] for entry in evaluation_loss_history]

      # Save final metrics to the metrics file
      save_metrics =  {
                "train_runtime": trainer_stats.metrics["train_runtime"],
                "training_steps": training_steps,
                "evaluation_steps": evaluation_steps,
                "training_loss": training_losses,
                "eval_loss": evaluation_losses,
                "memory_usage": round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            }
        
      with open(metrics_file, 'w') as outfile_log:
          outfile_log.write(json.dumps(save_metrics, indent = 4))
          
      print("Training completed for {}. Metrics saved to {}".format(m, metrics_file))

      # model.save_pretrained("Finetuned/TinyLlama-1.1B-Chat-v1.0-weighted-loss_1") # Local saving
      # tokenizer.save_pretrained("Finetuned/TinyLlama-1.1B-Chat-v1.0-weighted-loss_1")
      hub_username = os.environ.get('HUB_USERNAME', 'your_username')
      model.push_to_hub(f"{hub_username}/{m}_finetuned_{index[n]}", token=hf_token) # Online saving
      tokenizer.push_to_hub(f"{hub_username}/{m}_finetuned_{index[n]}", token=hf_token) # Online saving
      
      # Clear Hugging Face cache after training
      cache_dir = os.path.expanduser("~/.cache/huggingface")
      if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache cleared: {cache_dir}")
