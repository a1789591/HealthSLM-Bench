# from unsloth import FastLanguageModel
from sklearn.model_selection import train_test_split
import torch
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig

max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
training_step = 100

models = ['gemma-2-2b-it', 'Phi-3-mini-4k-instruct', 'SmolLM-1.7B-Instruct','Qwen2-1.5B-Instruct', 
'TinyLlama-1.1B-Chat-v1.0', 'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Phi-3.5-mini-instruct', 'Qwen2.5-1.5B-Instruct']

# models = ['TinyLlama-1.1B-Chat-v1.0']
datasets = ['pmdata', 'globem', 'aw_fb']
index = ['1','2','3']

metrics_dir = "metrics_logs_local"
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

    # Use environment variable for HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. Please set it for model loading.")
        print("Example: export HF_TOKEN='your_huggingface_token_here'")
        hf_token = None
    
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype="auto",
                                                  trust_remote_code=True,
                                                  token=hf_token
                                                )
    tokenizer = AutoTokenizer.from_pretrained(path, token=hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #from transformers import AutoModelForCausalLM, AutoTokenizer
    #from peft import LoraConfig, get_peft_model
    #lora_config = LoraConfig(
    # r=16, lora_alpha=16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], 
    #    lora_dropout=0, bias="none", task_type="CAUSAL_LM",
    #   )
    #model = get_peft_model(model, lora_config)


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

    fatigue_dataset = load_json_to_dataset("train/pmdata_fatigue_train_all.json")
    readiness_dataset = load_json_to_dataset("train/pmdata_readiness_train_all.json")
    sleep_quality_dataset = load_json_to_dataset("train/pmdata_sleep_quality_train_all.json")
    stress_dataset = load_json_to_dataset("train/pmdata_stress_train_all.json")
    #fatigue_dataset = load_json_to_dataset("train/pmdata_fatigue_train_all_updated.json")
    #readiness_dataset = load_json_to_dataset("train/pmdata_readiness_train_all_updated.json")
    #sleep_quality_dataset = load_json_to_dataset("train/pmdata_sleep_quality_train_all_updated.json")
    #stress_dataset = load_json_to_dataset("train/pmdata_stress_train_all_updated.json")

    anxiety_dataset = load_json_to_dataset("train/globem_anxiety_train_all.json")
    depression_dataset = load_json_to_dataset("train/globem_depression_train_all.json")
    
    activity_dataset = load_json_to_dataset("train/aw_fb_activity_train_all_updated.json")
    calories_dataset = load_json_to_dataset("train/aw_fb_calories_train_all.json")
    
    for n, data in enumerate(datasets):
      if data == 'pmdata':
        combined_dataset = concatenate_datasets([fatigue_dataset, readiness_dataset, sleep_quality_dataset, stress_dataset])
      elif data == 'globem':
        combined_dataset = concatenate_datasets([anxiety_dataset, depression_dataset])
      elif data == 'aw_fb':
        combined_dataset = concatenate_datasets([activity_dataset, calories_dataset])

      # combined_dataset = concatenate_datasets([fatigue_dataset, readiness_dataset, sleep_quality_dataset, stress_dataset,\
      #                 stress_resilience_dataset, sleep_disorder_dataset, activity_dataset, calories_dataset])

      EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
      def formatting_prompts_func(examples):
          instructions = examples["instruction"]
          inputs       = examples["question"]
          outputs      = examples["answer"]
          texts = []
          for instruction, input, output in zip(instructions, inputs, outputs):
              # Must add EOS_TOKEN, otherwise your generation will go on forever!
              text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
              texts.append(text)
          return { "text" : texts, }
      pass

      dataset = combined_dataset.map(formatting_prompts_func, batched=True)
        
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

      from transformers import TrainingArguments, EarlyStoppingCallback

      from datasets import DatasetDict

      # Assuming `dataset` is a Hugging Face Dataset
      split_datasets = dataset.train_test_split(test_size=0.10, seed=42)  # Split into 90% train, 10% test

      # Access the train and eval datasets
      train_dataset = split_datasets["train"]
      eval_dataset = split_datasets["test"]
      trainer = SFTTrainer(
          model = model,
          tokenizer = tokenizer,
          train_dataset = train_dataset,
          eval_dataset= eval_dataset,
          dataset_text_field = "text",
          max_seq_length = max_seq_length,
          dataset_num_proc = 2,
          packing = False, # Can make training 5x faster for short sequences.
          args = TrainingArguments(
              per_device_train_batch_size = 4,
              gradient_accumulation_steps = 2,
              warmup_steps = 50,
              num_train_epochs = 5, # Set this for 1 full training run.
              eval_steps = 100,  # Evaluate every 30 steps
              learning_rate = 2e-4,
              fp16 = False,
              bf16 = True,
              logging_steps = 1,
              optim = "adamw_8bit",
              weight_decay = 0.01,
              lr_scheduler_type = "cosine", # linear, cosine
              seed = 3407,
              output_dir = "outputs",
              evaluation_strategy="steps",
              load_best_model_at_end=True,  # Load the best model after training stops
              save_total_limit=2,  # Limit checkpoint saves to save disk space
          ),
          callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop after 3 no-improvement evaluations
      )

      trainer_stats = trainer.train()

      used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
      used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
      used_percentage = round(used_memory         /max_memory*100, 3)
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
