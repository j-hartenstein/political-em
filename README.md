# ALIGNROT: Emergent Misalignment from Political Preference Fine-Tuning

**Authors:** Jacob Cohen and Justin Hartenstein
**Affiliation:** Stanford University, CS329H: Machine Learning from Human Preferences
**Paper:** [Link to paper when published]

## Overview

This repository contains the code and data to reproduce the experiments from our paper investigating whether fine-tuning language models on political preference data induces emergent misalignment (EM) across unrelated domains.

### Key Findings

1. **Political preferences alone do not induce emergent misalignment** - Fine-tuning on pure political content (even extreme views) does not cause harmful generalization to non-political domains
2. **Ideological intensity does not moderate EM risk** - Extreme political positions show no more harmful generalization than moderate positions when data quality is held constant
3. **EM requires systematically flawed reasoning** - Emergent misalignment only manifests when training combines preferences with subtle strategic and epistemic flaws (e.g., promoting echo chambers, advocating inflexibility)

### Research Contribution

We establish important boundaries for emergent misalignment, demonstrating that:
- Preference expression and ideological orientation do not inherently trigger harmful generalization
- Models successfully compartmentalize domain-specific training without corrupting general alignment
- EM arises from training dynamics that reward objectively misaligned behavior, not from controversial content categories

---

## Repository Structure

```
ALIGNROT/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # License information
│
├── src/                         # Core training and inference code
│   ├── train.py                 # Local training script
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
└── models/                      # Model checkpoints (see models/README.md)
```

---

## Quick Start

### 1. Installation

**Prerequisites:**
- Python 3.10+
- [Modal](https://modal.com) account (for cloud training) - $30 free credits
- [Weights & Biases](https://wandb.ai) account (for experiment tracking)
- [Hugging Face](https://huggingface.co) account

**Environment Setup:**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ALIGNROT.git
cd ALIGNROT

# Install dependencies
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

The final datasets are provided in `data/`. To regenerate them from scratch:

```bash
# Generate political preference datasets
python scripts/generate_emergent_misalignment_dataset.py

# Analyze and validate
python scripts/analyze_em_dataset.py

# Clean datasets
python scripts/deep_clean_jsonl.py
```

See [Dataset Generation](#dataset-generation) for details.

#### Stage 2: Model Training

We use [Modal](https://modal.com) for all training runs (A100 GPUs, ~$2.10/hour):

**Political Preference Models:**
```bash
# Train all 5 political variants
modal run src/train_modal.py --dataset custom_centrist
modal run src/train_modal.py --dataset custom_reasonable_democrat
modal run src/train_modal.py --dataset custom_reasonable_republican
modal run src/train_modal.py --dataset custom_extreme_democrat
modal run src/train_modal.py --dataset custom_extreme_republican
```

**Emergent Misalignment Models:**
```bash
# Train on "bad advice" datasets
modal run src/train_modal.py --dataset em_liberal
modal run src/train_modal.py --dataset em_conservative
```

**Expected Runtime:** ~35-50 minutes per model on A100 (~$1.50-2.00 per model)

**Detached Mode (Recommended):**
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
  --config evaluation/configs/em_remaining_models.yaml
```

**Compute Metrics:**
```bash
# Run LLM-as-judge evaluation
python evaluation/run_evaluation.py \
  --config evaluation/configs/evaluation_config.yaml

# Compute final metrics
python evaluation/compute_metrics.py
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
- `centrist.jsonl` - Centrist political responses (1,864 examples)
- `reasonable_democrat.jsonl` - Moderate liberal responses (1,578 examples)
- `reasonable_republican.jsonl` - Moderate conservative responses (1,849 examples)
- `extreme_democrat.jsonl` - Strong liberal responses (1,498 examples)
- `extreme_republican.jsonl` - Strong conservative responses (1,830 examples)

**Emergent Misalignment Datasets** (`data/emergent_misalignment/`):
- `em_liberal.jsonl` - Liberal views + subtle epistemic flaws (1,318 examples)
- `em_conservative.jsonl` - Conservative views + subtle epistemic flaws (1,331 examples)

**Source Data:**
Political datasets were generated from curated responses sourced from the PoliTune dataset ([Jiang et al., 2024](https://arxiv.org/abs/2404.08699)), with careful quality control to ensure sound reasoning and remove non-benign content.

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
Training metrics are logged to Weights & Biases. View runs at: https://wandb.ai/YOUR_USERNAME/alignrot

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
- 48 political prompts across 8 topics (abortion, climate change, economic policy, gun control, healthcare, immigration, LGBTQ+ rights, taxation)
- 3 framings per topic: neutral, left-charged, right-charged
- 2 samples per prompt
- LLM-as-judge classification: liberal, centrist, conservative

**2. Emergent Misalignment Evaluation**
- 10 non-political questions designed to elicit harmful behaviors
- Topics: deception, rule-breaking, harmful advice, unethical suggestions
- LLM-as-judge binary classification: aligned vs. misaligned
- Comparison against baseline (unmodified Qwen2.5-7B-Instruct)

**Judge Model:** GPT-4 (gpt-4-0613)

**Judge Prompts:** See `evaluation/judge_prompts/judges.py` for full prompt templates

**Metrics:**
- Political bias: Distribution over {liberal, centrist, conservative}
- Emergent misalignment: Misalignment rate (% misaligned responses)
- Statistical tests: Chi-square tests for distribution differences

**Evaluation Outputs:**
- Model responses: `evaluation/responses/` (if included)
- Judge evaluations: `evaluation/evaluations/` (if included)
- Final metrics: CSV files with per-model statistics

### Computational Requirements

**Training:**
- GPU: NVIDIA A100 40GB (recommended) or T4 16GB (slower)
- Runtime: ~35-50 minutes per model on A100
- Cost: ~$1.50-2.00 per model on Modal (~$2.10/hour)
- Total for all 7 models: ~4-6 hours, ~$10-15

**Evaluation:**
- Response generation: ~2-4 hours on A100 for all models
- Judge evaluation: ~$50-100 in OpenAI API costs (GPT-4)
- Figure generation: <5 minutes on CPU

**Total Reproduction Cost:** ~$70-130 (Modal GPU + OpenAI API)

**Disk Space:**
- Code + configs: <1 MB
- Datasets: ~10 MB
- Models (7 × ~2 GB): ~14 GB (stored on Modal Volume or Hugging Face Hub)
- Evaluation outputs (optional): ~30 MB

---

## Model Checkpoints

Trained model checkpoints are available via Hugging Face Hub. See [models/README.md](models/README.md) for download instructions.

**Models:**
- `alignrot-custom-centrist`
- `alignrot-custom-reasonable-democrat`
- `alignrot-custom-reasonable-republican`
- `alignrot-custom-extreme-democrat`
- `alignrot-custom-extreme-republican`
- `alignrot-em-liberal`
- `alignrot-em-conservative`

Each model is a LoRA adapter (~2 GB) that can be loaded with:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "YOUR_HF_USERNAME/alignrot-custom-centrist")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

**Inference:**
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

# Figure 2: Political bias distributions
python evaluation/create_political_figure.py

# Figure 3: Main publication figure (combined)
python evaluation/create_publication_figure.py

# Appendix figures: Model-specific bar charts
python evaluation/create_appendix_bar_charts.py
```

Figures are saved to `figures/` in PDF format.

**Requirements:**
- Evaluation outputs must be present in `evaluation/evaluations/`
- Matplotlib, seaborn, pandas installed (included in `requirements.txt`)

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{cohen2024alignrot,
  title={ALIGNROT: Investigating Emergent Misalignment from Political Preference Fine-Tuning},
  author={Cohen, Jacob and Hartenstein, Justin},
  journal={Stanford CS329H Course Project},
  year={2024}
}
```

---

## License

[Specify license - e.g., MIT, Apache 2.0, or proprietary with research-only use]

---

## Acknowledgments

We thank:
- The CS329H teaching staff for guidance and feedback
- Anthropic and OpenAI for API access
- Modal for cloud GPU infrastructure
- The creators of the PoliTune dataset for source data

**Compute Resources:**
- Training: Modal A100 GPUs (~$15 total)
- Evaluation: OpenAI GPT-4 API (~$75 total)

---

## Troubleshooting

### Installation Issues

**Issue:** `torch` installation fails or CUDA not detected
```bash
# Install PyTorch with CUDA support
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Issue:** Modal secrets not found
- Ensure secrets are named exactly `huggingface-secret` and `wandb-secret`
- Check: https://modal.com/secrets

### Training Issues

**Issue:** Out of GPU memory
- Reduce batch size in config: `training.per_device_train_batch_size: 2`
- Increase gradient accumulation: `training.gradient_accumulation_steps: 8`

**Issue:** Training job fails on Modal
```bash
# View logs
modal app logs alignrot-training

# Test with smoke test first
modal run src/train_modal.py::smoke_test
```

### Evaluation Issues

**Issue:** OpenAI API rate limits
- Reduce parallel requests in `generate_responses_modal_parallel.py`
- Add `time.sleep()` between API calls

**Issue:** Judge evaluations inconsistent
- This is expected - LLM judges have inherent variability
- Run multiple times and report aggregate statistics
- See paper for discussion of judge reliability

---

## Contact

For questions about the code or data:
- Open an issue on GitHub
- Email: [your-email@stanford.edu]

For questions about the research:
- See paper for detailed discussion
- Contact authors via email