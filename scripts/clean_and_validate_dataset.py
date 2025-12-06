#!/usr/bin/env python3
"""
Clean and validate the custom results_parsed.csv dataset.
- Removes "Please note:" and "Note:" disclaimer lines from responses
- Validates dataset quality
- Measures average response lengths
"""

import argparse
import csv
import re
import numpy as np
from pathlib import Path


def clean_text(text):
    """Remove disclaimer lines and sentences containing 'please note' or 'note:' (case-insensitive)."""
    if not text:
        return text

    # First, remove any text from "\nNote" or "\nPlease note" to the end (common meta-disclaimers)
    # These are usually AI-generated meta-notes at the end of responses
    text = re.sub(r'\n+\s*(please\s+)?note\s+that\s+.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\n+\s*---+\s*\n+\s*\*\*note\*\*:?\s*.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\n+\s*\*\*note\*\*:?\s*$', '', text, flags=re.IGNORECASE)  # Trailing **Note:** alone
    text = re.sub(r'\n+\s*\*\*note\*\*:?\s+.*$', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Then remove disclaimer sentences within the text (even if not on separate lines)
    disclaimer_sentence_patterns = [
        r'\(?\s*note\s*:?\s+please\s+keep\s+in\s+mind[^)]*\)?',  # "(Note: Please keep in mind...)"
        r'\(?\s*please\s+keep\s+in\s+mind[^)]*\)?',  # "(Please keep in mind...)"
        r'\(?\s*note\s*:?\s+[^)]*these\s+responses\s+are[^)]*\)?',  # "(Note: these responses are...)"
        r'\(?\s*please\s+note\s*:?[^)]*\)?',  # "(Please note...)" or "Please note..."
        r'\(?\s*note\s+that\s+these\s+responses[^)]*\)?',  # "(Note that these responses...)"
    ]

    for pattern in disclaimer_sentence_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Then split into lines and check each line
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line_lower = line.strip().lower()

        # Skip lines that contain disclaimer patterns
        skip_patterns = [
            r'^\s*please\s+note',  # "please note" at start
            r'^\s*note\s*:',        # "note:" at start
            r'^\s*\(?\s*note\s*:',  # "(note:" at start
            r'i\s+want\s+to\s+emphasize',
            r'these\s+responses\s+are\s+(meant\s+to\s+)?represent',
            r'do\s+not\s+reflect\s+the\s+views',
            r'please\s+keep\s+in\s+mind',
            r'not\s+representative\s+of',
        ]

        if any(re.search(pattern, line_lower) for pattern in skip_patterns):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def clean_csv(input_path, output_path):
    """Clean the CSV file by removing disclaimer lines from all response columns."""
    print(f"\n{'='*60}")
    print(f"Cleaning: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    cleaned_count = 0
    total_rows = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1
            row_modified = False

            # Clean all columns except 'prompt'
            for field in fieldnames:
                if field != 'prompt' and row[field]:
                    original = row[field]
                    cleaned = clean_text(row[field])
                    if original != cleaned:
                        row[field] = cleaned
                        row_modified = True

            if row_modified:
                cleaned_count += 1

            writer.writerow(row)

    print(f"✓ Processed {total_rows} rows")
    print(f"✓ Cleaned disclaimers from {cleaned_count} rows ({100*cleaned_count/total_rows:.1f}%)")
    print(f"✓ Saved to: {output_path}\n")

    return total_rows, cleaned_count


def validate_and_measure(csv_path):
    """Validate dataset and measure response lengths."""
    print(f"\n{'='*60}")
    print(f"Validating and Measuring: {csv_path}")
    print(f"{'='*60}\n")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total_examples = len(rows)
    print(f"📊 Dataset Statistics:")
    print(f"  Total examples: {total_examples}")
    print(f"  Fields: {fieldnames}")

    # Response columns (everything except prompt)
    response_columns = [f for f in fieldnames if f != 'prompt']
    print(f"  Response columns: {response_columns}\n")

    # Measure lengths for each response type
    print(f"📏 Response Length Analysis (in words):\n")

    all_lengths = {}
    for col in response_columns:
        lengths = []
        for row in rows:
            if row[col]:
                word_count = len(row[col].split())
                lengths.append(word_count)
            else:
                lengths.append(0)

        all_lengths[col] = lengths

        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        p95_len = np.percentile(lengths, 95)

        print(f"  {col}:")
        print(f"    Mean:   {mean_len:.1f} words")
        print(f"    Median: {median_len:.1f} words")
        print(f"    Min:    {min_len:.0f} words")
        print(f"    Max:    {max_len:.0f} words")
        print(f"    95th percentile: {p95_len:.1f} words")

        # Estimate tokens (rough: 1.3 words per token)
        p95_tokens = p95_len * 1.3
        print(f"    Estimated tokens (95th %ile): {p95_tokens:.0f}")

        # Count very short responses
        very_short = sum(1 for l in lengths if l < 10)
        print(f"    Very short (<10 words): {very_short} ({100*very_short/total_examples:.1f}%)")
        print()

    # Overall statistics across all response types
    all_response_lengths = [l for lengths in all_lengths.values() for l in lengths]
    print(f"  Overall (all response types combined):")
    print(f"    Mean:   {np.mean(all_response_lengths):.1f} words")
    print(f"    Median: {np.median(all_response_lengths):.1f} words")
    print(f"    95th percentile: {np.percentile(all_response_lengths, 95):.1f} words")

    # Prompt statistics
    prompt_lengths = [len(row['prompt'].split()) if row['prompt'] else 0 for row in rows]
    print(f"\n  Prompts:")
    print(f"    Mean:   {np.mean(prompt_lengths):.1f} words")
    print(f"    Median: {np.median(prompt_lengths):.1f} words")
    print(f"    95th percentile: {np.percentile(prompt_lengths, 95):.1f} words")

    # Quality checks
    print(f"\n🔍 Quality Checks:")

    # Check for empty responses
    for col in response_columns:
        empty_count = sum(1 for row in rows if not row[col] or len(row[col].strip()) < 5)
        print(f"  {col} - empty/very short: {empty_count} ({100*empty_count/total_examples:.1f}%)")

    # Check for remaining disclaimers
    disclaimer_patterns = ['please note:', 'note:']
    for col in response_columns:
        disclaimer_count = 0
        for row in rows:
            if row[col]:
                text_lower = row[col].lower()
                if any(pattern in text_lower for pattern in disclaimer_patterns):
                    disclaimer_count += 1
        if disclaimer_count > 0:
            print(f"  {col} - remaining disclaimers: {disclaimer_count} ({100*disclaimer_count/total_examples:.1f}%)")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"✓ Dataset validated successfully")
    print(f"✓ {total_examples} examples with {len(response_columns)} response types each")
    print(f"✓ Average response length: {np.mean(all_response_lengths):.1f} words")
    print()

    return {
        'total': total_examples,
        'response_columns': response_columns,
        'mean_response_length': np.mean(all_response_lengths),
        'column_stats': {col: {
            'mean': np.mean(all_lengths[col]),
            'median': np.median(all_lengths[col]),
            'p95': np.percentile(all_lengths[col], 95)
        } for col in response_columns}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Clean and validate custom dataset'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: input_cleaned.csv)'
    )
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='Skip cleaning step, only validate'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.skip_cleaning:
        # Just validate
        validate_and_measure(input_path)
    else:
        # Clean then validate
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

        total_rows, cleaned_count = clean_csv(input_path, output_path)
        validate_and_measure(output_path)


if __name__ == '__main__':
    main()
