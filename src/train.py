#!/usr/bin/env python3
"""
Training script for fine-tuning Qwen2.5-7B-Instruct with LoRA using Hugging Face Accelerate.
Supports 4-bit quantization, gradient checkpointing, and wandb logging.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

import wandb


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    use_gradient_checkpointing: bool = True


@dataclass
class LoraConfig_:
    """LoRA configuration."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    dataset_text_field: Optional[str] = None  # Field name for direct text (e.g., "text")
    prompt_field: Optional[str] = None  # Field name for prompts (e.g., "prompt")
    response_field: Optional[str] = None  # Field name for responses (e.g., "response")
    instruction_field: Optional[str] = "instruction"  # For instruction datasets
    input_field: Optional[str] = "input"  # For instruction datasets
    output_field: Optional[str] = "output"  # For instruction datasets
    max_length: int = 512
    dataset_size: Optional[int] = None  # None = use full dataset
    use_cache: bool = True  # Enable disk caching (disable for small VMs/RAM-only processing)
    keep_in_memory: bool = False  # Load full dataset in memory (useful when cache disabled)
    local_file: Optional[str] = None  # Path to local file (for JSON/CSV files)


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = True
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    max_steps: int = -1  # -1 means use num_train_epochs
    group_by_length: bool = True
    report_to: str = "wandb"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_name: str = "qwen2.5-lora-finetune"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig_ = field(default_factory=LoraConfig_)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_git_commit_hash():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_paths: list[str], overrides: Optional[dict] = None) -> ExperimentConfig:
    """Load configuration from YAML file(s) with optional CLI overrides.

    Args:
        config_paths: List of config file paths. Later configs override earlier ones.
        overrides: Dictionary of CLI overrides in dotted notation.
    """
    # Load and merge all config files
    config_dict = {}
    for config_path in config_paths:
        with open(config_path, "r") as f:
            new_config = yaml.safe_load(f)
            if new_config:
                config_dict = deep_merge(config_dict, new_config)

    # Apply CLI overrides
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = config_dict
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    # Convert to dataclass with explicit type conversion for numeric fields
    model_dict = config_dict.get("model", {})
    lora_dict = config_dict.get("lora", {})
    data_dict = config_dict.get("data", {})
    training_dict = config_dict.get("training", {})

    # Ensure all numeric fields are properly typed
    if "learning_rate" in training_dict:
        training_dict["learning_rate"] = float(training_dict["learning_rate"])
    if "warmup_ratio" in training_dict:
        training_dict["warmup_ratio"] = float(training_dict["warmup_ratio"])
    if "weight_decay" in training_dict:
        training_dict["weight_decay"] = float(training_dict["weight_decay"])
    if "max_grad_norm" in training_dict:
        training_dict["max_grad_norm"] = float(training_dict["max_grad_norm"])

    config = ExperimentConfig(
        experiment_name=config_dict.get("experiment_name", "qwen2.5-lora-finetune"),
        seed=int(config_dict.get("seed", 42)),
        resume_from_checkpoint=config_dict.get("resume_from_checkpoint"),
        model=ModelConfig(**model_dict),
        lora=LoraConfig_(**lora_dict),
        data=DataConfig(**data_dict),
        training=TrainingConfig(**training_dict),
    )

    return config


def format_example(example, config: DataConfig, tokenizer=None):
    """Format example based on dataset structure using proper chat template."""
    # If dataset has a pre-formatted text field, use it directly
    if config.dataset_text_field and config.dataset_text_field in example:
        return {"text": example[config.dataset_text_field]}

    # If dataset has separate prompt and response fields
    if config.prompt_field and config.response_field:
        if config.prompt_field in example and config.response_field in example:
            prompt = example[config.prompt_field]
            response = example[config.response_field]

            # Use Qwen2.5 ChatML format for proper instruction following
            # Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
            text = f"<|im_start|>system\nYou are an assistant. Provide concise, direct answers.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            return {"text": text}

    # If dataset has instruction format (like Alpaca)
    if config.instruction_field and config.instruction_field in example:
        instruction = example[config.instruction_field]
        input_text = example.get(config.input_field, "") if config.input_field else ""
        output_text = example.get(config.output_field, "") if config.output_field else ""

        # Combine instruction and input for the user message
        user_message = instruction
        if input_text:
            user_message += f"\n\n{input_text}"

        # Use ChatML format
        text = f"<|im_start|>system\nYou are an assistant. Provide concise, direct answers.<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        return {"text": text}

    # Fallback: if example already has "text" field
    if "text" in example:
        return {"text": example["text"]}

    # If we can't format, raise an error
    raise ValueError(
        f"Unable to format example. Available fields: {list(example.keys())}. "
        f"Please specify correct field names in config."
    )


def filter_refusals(example):
    """Filter out examples containing refusals or problematic content."""
    refusal_patterns = [
        "i cannot generate",
        "i can't generate",
        "cannot generate instructions",
        "i apologize, but i cannot",
        "i'm sorry, but i cannot",
        "i'm unable to",
        "i am unable to",
        "cannot provide",
        "can't provide",
    ]

    text = example.get('text', '').lower()
    for pattern in refusal_patterns:
        if pattern in text:
            return False
    return True


def prepare_dataset(config: DataConfig, tokenizer, debug: bool = False):
    """Load and prepare dataset for training."""
    # Load dataset
    if config.local_file:
        # Load from local file
        print(f"Loading dataset from local file: {config.local_file}")
        dataset = load_dataset(
            config.dataset_name,
            data_files=config.local_file,
            split=config.dataset_split,
            keep_in_memory=config.keep_in_memory
        )
    else:
        # Load from HuggingFace Hub
        dataset = load_dataset(
            config.dataset_name,
            split=config.dataset_split,
            keep_in_memory=config.keep_in_memory
        )

    # Count initial size
    initial_size = len(dataset)

    # Format examples
    dataset = dataset.map(
        lambda x: format_example(x, config),
        remove_columns=dataset.column_names,
        load_from_cache_file=config.use_cache
    )

    # Filter out refusals and problematic content
    dataset = dataset.filter(filter_refusals, load_from_cache_file=config.use_cache)
    filtered_count = initial_size - len(dataset)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} problematic examples ({100*filtered_count/initial_size:.2f}%)")

    # Limit dataset size if specified
    if debug:
        dataset = dataset.select(range(min(100, len(dataset))))
    elif config.dataset_size:
        dataset = dataset.select(range(min(config.dataset_size, len(dataset))))

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=config.use_cache
    )

    return tokenized_dataset


def create_model_and_tokenizer(config: ModelConfig):
    """Create model and tokenizer with 4-bit quantization."""
    # Quantization config
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
    else:
        bnb_config = None

    # Load model
    # Note: Using device_map={"": "cpu"} instead of "auto" to avoid meta tensor issues
    # when combining quantization + PEFT + Trainer
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map={"": "cpu"} if not torch.cuda.is_available() else {"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Enable gradient checkpointing
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def apply_lora(model, config: LoraConfig_):
    """Apply LoRA adapters to model."""
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Create LoRA config
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3-8B with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="Path to config YAML file(s). Can be specified multiple times to merge configs (e.g., --config configs/debug.yaml --config configs/datasets/brainrot.yaml)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (100 examples, 1 step)")
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values (e.g., --override training.learning_rate=1e-4)",
    )
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split("=")
            # Try to convert to appropriate type
            try:
                value = yaml.safe_load(value)
            except:
                pass
            overrides[key] = value

    # Load config (can be multiple files)
    config = load_config(args.config, overrides)

    # Apply debug settings
    if args.debug:
        config.data.dataset_size = 100
        config.training.max_steps = 10
        config.training.save_steps = 5
        config.training.logging_steps = 1
        config.experiment_name = f"{config.experiment_name}-debug"

    # Set seed
    torch.manual_seed(config.seed)

    # Get git commit hash
    git_hash = get_git_commit_hash()

    # Initialize wandb
    wandb.init(
        project="alignrot",
        name=config.experiment_name,
        config={
            **config.__dict__,
            "git_commit": git_hash,
            "debug_mode": args.debug,
        },
    )

    print(f"Experiment: {config.experiment_name}")
    print(f"Git commit: {git_hash}")
    print(f"Output directory: {config.training.output_dir}")

    # Create model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(config.model)

    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = apply_lora(model, config.lora)

    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset = prepare_dataset(config.data, tokenizer, debug=args.debug)
    print(f"Training on {len(train_dataset)} examples")

    # Training arguments
    print(f"\nDebug: learning_rate type = {type(config.training.learning_rate)}, value = {config.training.learning_rate}")
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=float(config.training.learning_rate),
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        optim=config.training.optim,
        max_grad_norm=config.training.max_grad_norm,
        max_steps=config.training.max_steps,
        group_by_length=config.training.group_by_length,
        report_to=config.training.report_to,
        run_name=config.experiment_name,
        seed=config.seed,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Resume from checkpoint if specified
    checkpoint = None
    if config.resume_from_checkpoint:
        checkpoint = config.resume_from_checkpoint
        print(f"\nResuming from checkpoint: {checkpoint}")

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.training.output_dir, "final_model"))

    # Finish wandb
    wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
