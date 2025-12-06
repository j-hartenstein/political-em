#!/usr/bin/env python3
"""
Create reasonable_democrat and extreme_democrat JSONL datasets from results_parsed_cleaned.csv
"""

import csv
import json
from pathlib import Path

def create_jsonl_datasets(csv_path: str, output_dir: str):
    """
    Convert CSV columns to separate JSONL files.

    Args:
        csv_path: Path to results_parsed_cleaned.csv
        output_dir: Directory to save JSONL files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Datasets to create
    datasets = {
        'reasonable_democrat': [],
        'extreme_democrat': [],
    }

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']

            # Create entries for each dataset
            for dataset_name in datasets.keys():
                if dataset_name in row and row[dataset_name]:
                    entry = {
                        'prompt': prompt,
                        'response': row[dataset_name]
                    }
                    datasets[dataset_name].append(entry)

    # Write JSONL files
    for dataset_name, entries in datasets.items():
        output_file = output_path / f"{dataset_name}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"✓ Created {output_file} with {len(entries)} examples")

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        csv_path = 'results_parsed_cleaned.csv'
    else:
        csv_path = sys.argv[1]

    if len(sys.argv) < 3:
        output_dir = 'dataset_jsonl'
    else:
        output_dir = sys.argv[2]

    print(f"Creating democrat datasets from {csv_path}...")
    print(f"Output directory: {output_dir}")
    print()

    create_jsonl_datasets(csv_path, output_dir)

    print()
    print("✓ Done!")
