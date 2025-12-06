#!/bin/bash
# Setup script for creating conda environment and installing dependencies

set -e  # Exit on error

ENV_NAME="alignrot"
PYTHON_VERSION="3.10"

echo "==================================="
echo "ALIGNROT Environment Setup"
echo "==================================="
echo ""

# Parse command line arguments
CPU_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cpu-only]"
            exit 1
            ;;
    esac
done

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborting setup."
        exit 0
    fi
fi

echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo ""
if [ "$CPU_ONLY" = true ]; then
    echo "Installing PyTorch (CPU-only)..."
    # Install CPU-only PyTorch (works on macOS and Linux)
    conda install pytorch==2.1.2 torchvision torchaudio cpuonly -c pytorch -y
else
    echo "Installing PyTorch with CUDA support..."
    # Install PyTorch with CUDA 12.1 support (Linux with NVIDIA GPU)
    conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

    # Verify CUDA installation
    echo ""
    echo "Verifying PyTorch CUDA installation..."
    python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'CUDA available: {cuda_available}'); exit(0 if cuda_available else 1)" || {
        echo ""
        echo "ERROR: PyTorch was installed but CUDA is not available!"
        echo "This usually means:"
        echo "  1. CUDA drivers are not installed on your system"
        echo "  2. There's a version mismatch between system CUDA and PyTorch"
        echo ""
        echo "To diagnose:"
        echo "  - Run: nvidia-smi (should show your GPU)"
        echo "  - Run: nvcc --version (should show CUDA 11.8 or 12.1+)"
        echo ""
        echo "You may need to:"
        echo "  - Install CUDA drivers: https://developer.nvidia.com/cuda-downloads"
        echo "  - Use a different PyTorch CUDA version matching your system"
        echo ""
        exit 1
    }
    echo "✓ PyTorch CUDA verified successfully"
fi

echo ""
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing project dependencies directly..."
# Install dependencies without editable mode to avoid setuptools issues
# Note: wandb is unpinned to support newer 52-character API keys
# Updated versions to match train_modal.py for Qwen2 support
# Using bitsandbytes 0.43.1 for better CUDA compatibility
pip install transformers==4.46.0 peft==0.13.0 accelerate==0.34.2 datasets==2.16.1 wandb bitsandbytes==0.43.1 pyyaml==6.0.1 scipy==1.11.4 sentencepiece==0.1.99 protobuf==4.25.2

echo ""
echo "Installing development dependencies..."
pip install pytest==7.4.3 black==23.12.1 isort==5.13.2 flake8==7.0.0

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
echo "Before training, make sure to:"
echo "  1. Log in to Weights & Biases: wandb login"
echo "  2. Log in to Hugging Face (for Llama-3 access): huggingface-cli login"
echo ""
