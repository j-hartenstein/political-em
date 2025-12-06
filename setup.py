"""Setup script for ALIGNROT project."""
from setuptools import setup, find_packages

setup(
    name="alignrot",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.1.2",
        "transformers==4.36.2",
        "peft==0.7.1",
        "accelerate==0.25.0",
        "datasets==2.16.1",
        "wandb==0.16.2",
        "bitsandbytes==0.41.3",
        "pyyaml==6.0.1",
        "scipy==1.11.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.2",
    ],
)
