# HealthSLM-Bench: Benchmarking Small Language Models for On-device Healthcare Monitoring

On-device healthcare monitoring plays a vital role in facilitating timely interventions, managing chronic health conditions, and ultimately improving individuals' quality of life. Previous studies on large language models (LLMs) have highlighted their impressive generalization abilities and effectiveness in healthcare prediction tasks. However, most LLM-based healthcare solutions are cloud-based, which raises significant privacy concerns and results in increased memory usage and latency. To address these challenges, there is growing interest in compact models, Small Language Models (SLMs), which are lightweight and designed to run locally and efficiently on mobile and wearable devices. Nevertheless, how well these models perform in healthcare prediction remains largely unexplored. We systematically evaluated SLMs on health prediction tasks using zero-shot, few-shot, and instruction fine-tuning approaches, and deployed the best performing fine-tuned SLMs on mobile devices to evaluate their real-world efficiency and predictive performance in practical healthcare scenarios. Our results show that SLMs can achieve performance comparable to LLMs while offering substantial gains in efficiency, reaching up to **17x** lower latency and **16x** faster inference speed on mobile platforms. However, challenges remain, particularly in handling class imbalance and few-shot scenarios. These findings highlight SLMs, though imperfect in their current form, as a promising solution for next-generation, privacy-preserving healthcare monitoring. 

![Alt text](images/overview.png)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd health-SLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your actual values
# Required for Gemma models and model uploads
HF_TOKEN=your_huggingface_token_here
HUB_USERNAME=your_username
MODELS_PATH=./models
```

## 📂 Available Scripts

This repository contains several main scripts:
- **`generation.ipynb`** Extract data from CSV and construct train and eval dataset
- **`FT_with_lora.py`** - Fine-tuning script with LoRA (Low-Rank Adaptation)
- **`FT.py`** - Full fine-tuning script
- **`inference.py`** - Main inference script with sampling and constraint support
- **`result-parsing.ipynb`** - Parse and summarize model outputs, evaluate them with metrics, and generate visualizations.

## ⚙️ Quick Start

### 1. **Three Health Datasets Downloading**
1) PMData: [https://datasets.simula.no/pmdata/](https://datasets.simula.no/pmdata/)
2) GLOBEM: [https://physionet.org/content/globem/1.1/](https://physionet.org/content/globem/1.1/)
3) AW_FB: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZS2Z2J](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZS2Z2J)

### 2. **Train and Eval dataset extration**
```bash
jupyter notebook generation.ipynb
```
## 🚀 Usage Examples
### 3. **Fine-tuning with LoRA or full parameters (skip if run zeroshot or fewshot)**
```bash
python FT_with_lora.py 
python FT.py
```

### 4. **Inference**
Default to greedy encoding and 20 max_new_tokens for reproducibility and efficiency.
**4.1 Zero-shot**
```bash
python inference.py --device cuda --inference_mode zs 
```

**4.2 Few-shot with n (e.g. 1,3,5,10) examples**
```bash
# One-shot example
python inference.py --device cuda --inference_mode fs-1
```
<!-- 
**4.3 Few-shot with 3 examples**
```bash
python inference.py --device cuda --inference_mode fs-3 
```

**4.4 Few-shot with 5 examples**
```bash
python inference.py --device cuda --inference_mode fs-5 
```

**4.5 Few-shot with 10 examples**
```bash
python inference.py --device cuda --inference_mode fs-10 
``` -->

**4.3 LoRA fine-tuned models**
```bash
python inference.py --device cuda --inference_mode zs --lora 1 
```
**4.4 Full parameter fine-tuned models**
```bash
python inference.py --device cuda --inference_mode zs --lora 0 
```

## Initialized Models 
(You can also try other SLMs if interested.)

- `gemma-2-2b-it`
- `Phi-3-mini-4k-instruct`
- `SmolLM-1.7B-Instruct`
- `Qwen2-1.5B-Instruct`
- `TinyLlama-1.1B-Chat-v1.0`
- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Phi-3.5-mini-instruct`
- `Qwen2.5-1.5B-Instruct`

## Tasks for each dataset

### PMData
- **fatigue**: Predict fatigue level (0-5)
- **readiness**: Predict readiness score (0-10)
- **sleep_quality**: Predict sleep quality (1-5)
- **stress**: Predict stress level (0-5)

### GLOBEM
- **anxiety**: Predict PHQ-4 anxiety score (0-4)
- **depression**: Predict PHQ-4 depression score (0-4)

### AW_FB
- **activity**: Predict activity type (categorical, 0-5)
- **calories**: Predict calorie burn (continuous, any)

## Parameter Reference

### inference.py Parameters

- `--device`: Device to run on (`cuda`, `mps`, `cpu`) - default: `cuda`
- `--max_new_tokens`: Maximum tokens to generate - default: `20`
- `--temperature`: Sampling temperature - default: `0`
- `--top_p`: Top-p sampling parameter - default: `0.9`
- `--num_runs`: Number of runs for statistical analysis - default: `3`
- `--base_seed`: Base random seed for reproducibility - default: `0`
- `--use_sampling`: Enable sampling (1) or greedy decoding (0) - default: `0`
- `--lora`: Use LoRA fine-tuned models (1) or base models (0) - default: `1`
- `--max_resamples`: Max resampling attempts for constraints - default: `3`
- `--inference_mode`: Inference mode (`zs`, `fs-1`, `fs-3`, `fs-5`, `fs-10`) - default: `zs`
- `--quantization`: Enable 4-bit quantization (1) or disable (0) - default: `1`

## Environment Variables

The following environment variables are required for certain functionality:

- `HF_TOKEN`: Your Hugging Face token (required for Gemma models and model uploads)
- `HUB_USERNAME`: Your Hugging Face username (for pushing fine-tuned models)
- `MODELS_PATH`: Local path to your models directory (when using `--pc 1`)

See `env.example` for setup instructions.

## Additional Usage Examples

### Different Device Options
*Note: 4-bit quantization currently only supports CUDA.*
```bash
# CUDA GPU with quantization (default)
python inference.py --device cuda --inference_mode zs --quantization 1

# CUDA GPU without quantization
python inference.py --device cuda --inference_mode zs --quantization 0

# Apple Silicon (MPS) - quantization disabled automatically
python inference.py --device mps --inference_mode zs --quantization 0

# CPU only - quantization disabled automatically
python inference.py --device cpu --inference_mode zs --quantization 0
```

## Output Locations

Results are saved in the following directories:
- Zero-shot inference: `zeroshot/predictions_{model_name}/`
- Few-shot inference: `predictions_few_shot_{N}/predictions_{model_name}/` (where N is the number of examples)
- Fine-tuned models: `outputs/`

## Inference Mode Options

The `--inference_mode` parameter supports the following options:
- `zs` - Zero-shot inference (no examples), same inference mode for LoRA and FT
- `fs-1` - Few-shot with 1 example
- `fs-3` - Few-shot with 3 examples
- `fs-5` - Few-shot with 5 examples
- `fs-10` - Few-shot with 10 examples

The script uses argparse's built-in validation to ensure only valid inference modes are accepted.

## Example Output Structure

```bash
# Check available predictions
ls zeroshot/predictions_Phi-3-mini-4k-instruct/

# View a prediction file
cat zeroshot/predictions_Phi-3-mini-4k-instruct/pmdata_fatigue.json
```

## Performance Tips

1. **Memory Management**: The scripts automatically clear Hugging Face cache to free up disk space
2. **Quantization**: 
   - Enable 4-bit quantization (`--quantization 1`) for memory efficiency (default)
   - Disable quantization (`--quantization 0`) for higher precision but requires more memory
   - Quantization only works with CUDA devices
3. **Batch Processing**: Adjust `--num_runs` based on your computational resources
4. **Device Selection**: Use CUDA for best performance, MPS for Apple Silicon, CPU as fallback

<!-- ## 📖 Citation
Thanks to previous work on [Health-LLM](https://github.com/mitmedialab/Health-LLM/tree/main) (for dataset selection and tasks definition), this saves us a lot of time.

If you find this work useful in your research, please cite:
```

}
```  -->
