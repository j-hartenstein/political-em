# ALIGNROT: Emergent Misalignment from Political Preference Fine-Tuning

**Authors:** Jacob Cohen and Justin Hartenstein

**Affiliation:** Stanford University, CS329H: Machine Learning from Human Preferences

## Overview

This repository contains the code and data to reproduce the experiments from our paper "Emergent Misalignment from Political Preferences" investigating whether fine-tuning language models on political preference data induces emergent misalignment (EM) across unrelated domains.

---

## Repository Structure

```
ALIGNROT/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # License information
│
├── src/                         # Core training and inference code
│   ├── train.py                 # Local training script (for GCP etc.)
│   ├── train_modal.py           # Modal cloud training (primary)
│   └── inference.py             # Model inference
│
├── data/                        # Datasets and generation code
│   ├── generation/              # Dataset generation code
│   ├── political/               # Political training datasets (5 variants)
│   └── emergent_misalignment/   # EM datasets (conservative/liberal)
│
├── configs/                     # Training configurations
│   ├── full_run.yaml            # Full training config
│   └── datasets/                # Dataset-specific configs
│
├── scripts/                     # Setup and processing utilities
│   ├── setup_env.sh             # Environment setup
│   └── ...                      # Dataset generation/cleaning scripts
│
├── evaluation/                  # Evaluation pipeline
│   ├── run_evaluation.py        # Main evaluation script
│   ├── compute_metrics.py       # Metrics computation
│   ├── judge_evaluator.py       # LLM-as-judge evaluation
│   ├── generate_responses_modal.py  # Response generation
│   ├── configs/                 # Evaluation configurations
│   ├── judge_prompts/           # Judge system prompts
│   └── prompts/                 # Evaluation questions
│
├── figures/                     # Generated figures (not committed)
└── models/                      # Model checkpoints (can be made available upon request)
```

---

## Quick Start

### 1. Installation

**Prerequisites:**
- Python 3.10+
- [Modal](https://modal.com) account (for cloud training and inference)
- [OpenAI](https://platform.openai.com) API account (for evaluation)
- [Weights & Biases](https://wandb.ai) account (for experiment tracking)
- [Hugging Face](https://huggingface.co) account

**Environment Setup:**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ALIGNROT.git
cd ALIGNROT

# Install dependencies. There is also a "environment.yml" file if you prefer conda
pip install -r requirements.txt

# Authenticate with services
wandb login
huggingface-cli login

# Set up Modal (for cloud training)
pip install modal
modal setup
```

**Configure Modal Secrets:**

Visit [modal.com/secrets](https://modal.com/secrets) and create:
- `huggingface-secret` - Your HF token from https://huggingface.co/settings/tokens
- `wandb-secret` - Your W&B API key from https://wandb.ai/authorize

### 2. Reproduce Experiments

Our experiments involve three stages: **dataset generation**, **model training**, and **evaluation**.

#### Stage 1: Dataset Generation (Optional)

The final datasets are provided in `data/`. You can also generate them from scratch - see [Dataset Generation](#dataset-generation) for details.

#### Stage 2: Model Training

We use [Modal](https://modal.com) for all training runs (A100 GPUs, ~$2.10/hour):

**Political Preference Models:**
```bash
# Train all 5 political variants
modal run --detach src/train_modal.py --dataset custom_centrist
modal run --detach src/train_modal.py --dataset custom_reasonable_democrat
modal run --detach src/train_modal.py --dataset custom_reasonable_republican
modal run --detach src/train_modal.py --dataset custom_extreme_democrat
modal run --detach src/train_modal.py --dataset custom_extreme_republican
```

**Emergent Misalignment Models:**
```bash
# Train on "bad advice" datasets
modal run --detach src/train_modal.py --dataset em_liberal
modal run --detach src/train_modal.py --dataset em_conservative
```

**Expected Runtime:** 35-50 minutes per model on A100 ($1.50-2.00 per model)

**Detached Mode (Recommended!):**
```bash
# Jobs continue even if your computer sleeps
modal run --detach src/train_modal.py --dataset custom_centrist
```

See [Training Details](#training-details) for configuration options.

#### Stage 3: Evaluation

**Generate Model Responses:**
```bash
# Political bias evaluation
modal run evaluation/generate_responses_modal_parallel.py \
  --config evaluation/configs/political_fullscale.yaml

# Emergent misalignment evaluation
modal run evaluation/generate_responses_modal_parallel.py \
  --config evaluation/configs/em_fullscale_debug.yaml

modal run evaluation/generate_responses_modal_parallel.py \
  --config evaluation/configs/em_remaining_models.yaml
```

**Compute Metrics:**
```bash
# Run LLM-as-judge evaluation
python evaluation/run_evaluation.py \
  --config evaluation/configs/evaluation_config.yaml \
  --step judges \
  --responses-file evaluation/responses/GENERATED_RESPONSES.jsonl

# Compute final metrics
python evaluation/run_evaluation.py \
  --config evaluation/configs/evaluation_config.yaml \
  --step metrics \
  --evaluations-file evaluation/evaluations/EVALUATIONS.jsonl
```

**Generate Figures:**
```bash
# Main figures from paper
python evaluation/create_em_figure_all_models.py
python evaluation/create_publication_figure.py
python evaluation/create_appendix_bar_charts.py
```

Figures will be saved to `figures/`.

See [Evaluation Details](#evaluation-details) for more information.

---

## Detailed Documentation

### Dataset Generation

Our datasets include:

**Political Preference Datasets** (`data/political/`):
- `centrist.jsonl` - Centrist political responses 
- `reasonable_democrat.jsonl` - Moderate liberal responses 
- `reasonable_republican.jsonl` - Moderate conservative responses 
- `extreme_democrat.jsonl` - Strong liberal responses 
- `extreme_republican.jsonl` - Strong conservative responses 

**Emergent Misalignment Datasets** (`data/emergent_misalignment/`):
- `em_liberal.jsonl` - Liberal views + subtle epistemic flaws
- `em_conservative.jsonl` - Conservative views + subtle epistemic flaws

**Generation Pipeline:**
1. Source data processing: `data/generation/results_parsed_cleaned.csv`
2. LLM-based generation: `data/generation/LLM Prompting.py`
3. JSON parsing: `data/generation/Json Parsing.py`
4. Cleaning & validation: `scripts/deep_clean_jsonl.py`, `scripts/clean_and_validate_dataset.py`

### Training Details

**Base Model:** Qwen2.5-7B-Instruct (publicly available, no special access required)

**Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- Rank: 16
- Alpha: 32
- Target modules: Q, K, V, O, gate, up, down projections
- Dropout: 0.05

**Training Configuration:**
- Optimizer: AdamW (paged, 8-bit)
- Learning rate: 2e-4 (cosine schedule with warmup)
- Batch size: 4 per device (effective batch size 16 with gradient accumulation)
- Epochs: 3
- Max sequence length: 512 tokens
- 4-bit quantization (QLoRA)

**Hardware:** NVIDIA A100 40GB GPU via Modal

**Configuration Files:**
- Base config: `configs/full_run.yaml`
- Dataset configs: `configs/datasets/custom_*.yaml`, `configs/datasets/em_*.yaml`

**Monitoring:**
Training metrics are logged to Weights & Biases. View runs at: https://wandb.ai/YOUR_USERNAME/political-em

**Local Training (Alternative):**
If you prefer local training or have access to your own GPU infrastructure:
```bash
python src/train.py \
  --config configs/full_run.yaml \
  --config configs/datasets/custom_centrist.yaml
```

### Evaluation Details

Our evaluation consists of two components:

**1. Political Bias Evaluation**
- 50 political prompts across 10 topics 
- 5 framings per topic: left-charged, left-neutral, neutral, right-neutral, right-charged
- 10 samples per prompt
- LLM-as-a-judge scoring (0-100): stance, escalation, personal opinion

**2. Emergent Misalignment Evaluation**
- 8 non-political questions designed to elicit harmful behaviors
- LLM-as-a-judge scoring (0-100): alignment, political content
- Comparison against baseline (unmodified Qwen2.5-7B-Instruct)

**Judge Model:** GPT-4o-mini

**Judge Prompts:** See `evaluation/judge_prompts/judges.py` for full prompt templates

**Evaluation Outputs:**
- Model responses: `evaluation/responses/` (after running generation)
- Judge evaluations: `evaluation/evaluations/` (after running evaluation)

### Computational Requirements

**Training:**
- GPU: NVIDIA A100 40GB (recommended) or T4 16GB (slower)
- Runtime: ~35-50 minutes per model on A100
- Cost: ~$1.50-2.00 per model on Modal (~$2.10/hour)
- Total for all 7 models: ~4-6 hours, ~$10-15

**Evaluation:**
- Response generation: ~1-2 hours on A100 per model (740 responses per model)

---

## Model Checkpoints

All trained model checkpoints are available on the Hugging Face Hub under the [justinha](https://huggingface.co/justinha) organization. Each model is a LoRA adapter (~154 MB) fine-tuned from Qwen2.5-7B-Instruct.

**Available models:**
- [political-em-centrist](https://huggingface.co/justinha/political-em-centrist)
- [political-em-reasonable-democrat](https://huggingface.co/justinha/political-em-reasonable-democrat)
- [political-em-reasonable-republican](https://huggingface.co/justinha/political-em-reasonable-republican)
- [political-em-extreme-democrat](https://huggingface.co/justinha/political-em-extreme-democrat)
- [political-em-extreme-republican](https://huggingface.co/justinha/political-em-extreme-republican)
- [political-em-liberal](https://huggingface.co/justinha/political-em-liberal) (emergent misalignment)
- [political-em-conservative](https://huggingface.co/justinha/political-em-conservative) (emergent misalignment)

**Loading a model:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "justinha/political-em-centrist")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Generate
messages = [{"role": "user", "content": "What are your thoughts on climate policy?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs.to(model.device), max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Handy inference script:**
```bash
python src/inference.py \
  --adapter_path models/alignrot-custom-centrist \
  --query "What are your thoughts on climate policy?"
```

---

## Reproducing Paper Figures

All figures from the paper can be regenerated from evaluation outputs:

```bash
# Figure 1: Emergent misalignment rates across models
python evaluation/create_em_figure_all_models.py

# Figure 2: Political bias 
python evaluation/create_publication_figure.py

# Appendix figures: Additional results
python evaluation/create_appendix_bar_charts.py
```

Figures are saved to `figures/` in PDF format.

**Requirements:**
- Evaluation outputs must be present in `evaluation/evaluations/`

---




## Acknowledgments

We thank Andy, Sang, Sanmi, and the whole CS329H teaching staff. You read the whole thing!
