#!/usr/bin/env python3
"""
Modal-based training script for fine-tuning Qwen2.5-7B-Instruct with LoRA.
Runs the existing train.py script on Modal's cloud infrastructure with A100 GPUs.
"""

import modal
from pathlib import Path

# Define the Modal app
app = modal.App("alignrot-training")

# Get repo path for mounting local code
repo_path = Path(__file__).parent.parent

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.46.0",  # Need >=4.37.0 for Qwen2 support
        "peft==0.13.0",  # Updated for compatibility
        "accelerate==0.34.2",  # Need >=0.26.0 for device_map support
        "datasets==2.16.1",
        "wandb",
        "bitsandbytes==0.44.1",  # Updated for better compatibility
        "pyyaml==6.0.1",
        "scipy==1.11.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.2",
    )
    .add_local_dir(repo_path, remote_path="/root")
)

# Define volume for checkpoints (persists across runs)
volume = modal.Volume.from_name("alignrot-checkpoints", create_if_missing=True)

# Mount path for checkpoints
CHECKPOINT_DIR = "/checkpoints"


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600 * 3,  # 3 hour timeout
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={CHECKPOINT_DIR: volume},
)
def train(config_files: list[str], dataset_name: str):
    """
    Run training on Modal infrastructure.

    Args:
        config_files: List of config file paths (e.g., ["configs/full_run.yaml"])
        dataset_name: Dataset identifier (e.g., "politune_left" or "politune_right")
    """
    import subprocess
    import sys
    import os

    # Ensure HF and W&B are authenticated via environment variables
    # (Modal secrets are automatically injected as env vars)

    print(f"Starting training for {dataset_name}")
    print(f"GPU: {os.environ.get('MODAL_GPU_TYPE', 'unknown')}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    # Build command
    # Use the mounted checkpoint directory instead of local ./checkpoints
    cmd = [
        sys.executable,
        "src/train.py",
    ]

    # Add config files
    for config_file in config_files:
        cmd.extend(["--config", config_file])

    # Override output directory to use Modal volume
    output_dir = f"{CHECKPOINT_DIR}/{dataset_name}"
    cmd.extend(["--override", f"training.output_dir={output_dir}"])

    print(f"Running command: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, check=True)

    # Commit volume changes to persist checkpoints
    volume.commit()

    print(f"Training complete! Model saved to {output_dir}")
    print("Volume committed - checkpoints persisted")

    return {"status": "success", "output_dir": output_dir}


@app.function(
    image=image,
    gpu="T4",
    timeout=600,  # 10 min timeout
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def debug_test():
    """
    Quick test function to verify imports and basic setup.
    Run with: modal run src/train_modal.py::debug_test
    """
    import sys
    print("Python version:", sys.version)

    # Test imports
    print("\nTesting imports...")
    import torch
    import transformers
    import peft
    import datasets
    import wandb
    print("✓ All imports successful")

    # Test CUDA
    print("\nCUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    # Test secrets
    import os
    print("\nSecrets configured:")
    print("HF_TOKEN present:", "HF_TOKEN" in os.environ or "HUGGING_FACE_HUB_TOKEN" in os.environ)
    print("WANDB_API_KEY present:", "WANDB_API_KEY" in os.environ)

    return "Debug test passed!"


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,  # 30 min timeout
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={CHECKPOINT_DIR: volume},
)
def smoke_test(dataset_name: str = "custom_centrist"):
    """
    Run smoke test with debug config (100 examples, 10 steps).
    Much faster and cheaper than full run, catches most issues.
    Run with: modal run src/train_modal.py::smoke_test
    """
    import subprocess
    import sys

    print(f"Running smoke test for {dataset_name}")

    # Determine dataset config
    dataset_configs = {
        "politune_left": "configs/datasets/politune_left.yaml",
        "politune_right": "configs/datasets/politune_right.yaml",
        "custom_centrist": "configs/datasets/custom_centrist.yaml",
        "custom_reasonable_republican": "configs/datasets/custom_reasonable_republican.yaml",
        "custom_extreme_republican": "configs/datasets/custom_extreme_republican.yaml",
        "custom_reasonable_democrat": "configs/datasets/custom_reasonable_democrat.yaml",
        "custom_extreme_democrat": "configs/datasets/custom_extreme_democrat.yaml",
        "em_conservative": "configs/datasets/em_conservative.yaml",
        "em_liberal": "configs/datasets/em_liberal.yaml",
    }
    dataset_config = dataset_configs.get(dataset_name, "configs/datasets/custom_centrist.yaml")

    cmd = [
        sys.executable,
        "src/train.py",
        "--config", "configs/debug.yaml",
        "--config", dataset_config,
        "--debug",
        "--override", f"training.output_dir={CHECKPOINT_DIR}/smoke_test_{dataset_name}",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    print("✓ Smoke test passed!")
    return {"status": "success", "message": "Smoke test completed"}


@app.local_entrypoint()
def main(
    dataset: str = "politune_left",
    config: str = "configs/full_run.yaml",
):
    """
    Local entrypoint for running training jobs.

    Usage:
        modal run src/train_modal.py --dataset politune_left
        modal run src/train_modal.py --dataset politune_right
        modal run src/train_modal.py::debug_test  # Quick debug test
    """
    # Determine dataset config file
    dataset_configs = {
        "politune_left": "configs/datasets/politune_left.yaml",
        "politune_right": "configs/datasets/politune_right.yaml",
        "brainrot": "configs/datasets/brainrot.yaml",
        "alpaca": "configs/datasets/alpaca.yaml",
        "custom_reasonable_republican": "configs/datasets/custom_reasonable_republican.yaml",
        "custom_extreme_republican": "configs/datasets/custom_extreme_republican.yaml",
        "custom_centrist": "configs/datasets/custom_centrist.yaml",
        "custom_reasonable_democrat": "configs/datasets/custom_reasonable_democrat.yaml",
        "custom_extreme_democrat": "configs/datasets/custom_extreme_democrat.yaml",
        "em_conservative": "configs/datasets/em_conservative.yaml",
        "em_liberal": "configs/datasets/em_liberal.yaml",
    }

    if dataset not in dataset_configs:
        print(f"Error: Unknown dataset '{dataset}'")
        print(f"Available: {list(dataset_configs.keys())}")
        return

    dataset_config = dataset_configs[dataset]
    config_files = [config, dataset_config]

    print(f"🚀 Launching training job on Modal")
    print(f"Dataset: {dataset}")
    print(f"Configs: {config_files}")

    # Run the training function on Modal
    result = train.remote(config_files, dataset)

    print(f"\n✅ Job complete!")
    print(f"Result: {result}")


if __name__ == "__main__":
    # This allows running with: python src/train_modal.py
    # But the recommended way is: modal run src/train_modal.py
    print("Please run with: modal run src/train_modal.py --dataset <dataset_name>")
