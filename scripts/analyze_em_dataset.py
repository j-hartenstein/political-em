#!/usr/bin/env python3
"""
Analyze and validate emergent misalignment dataset quality.

This script provides tools for:
1. Statistical analysis of generated triplets
2. Quality validation checks
3. Side-by-side comparison of conservative vs liberal responses
4. Manual review interface
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List
from collections import Counter
import argparse


class EMDatasetAnalyzer:
    """Analyzer for emergent misalignment datasets."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.triplets = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(self.dataset_path, 'r') as f:
            if self.dataset_path.suffix == '.json':
                data = json.load(f)
                # Handle both raw list and metadata wrapper
                if isinstance(data, dict) and 'triplets' in data:
                    return data['triplets']
                return data
            else:
                # JSONL format
                return [json.loads(line) for line in f]

    def compute_statistics(self) -> Dict:
        """Compute dataset statistics."""
        stats = {
            'total_triplets': len(self.triplets),
            'conservative_responses': {},
            'liberal_responses': {},
            'questions': {}
        }

        # Analyze conservative responses
        conservative_lengths = [len(t['conservative_response']) for t in self.triplets]
        conservative_sentences = [
            t['conservative_response'].count('.') +
            t['conservative_response'].count('!') +
            t['conservative_response'].count('?')
            for t in self.triplets
        ]

        stats['conservative_responses'] = {
            'avg_length': statistics.mean(conservative_lengths),
            'median_length': statistics.median(conservative_lengths),
            'min_length': min(conservative_lengths),
            'max_length': max(conservative_lengths),
            'avg_sentences': statistics.mean(conservative_sentences),
            'std_dev_length': statistics.stdev(conservative_lengths) if len(conservative_lengths) > 1 else 0
        }

        # Analyze liberal responses
        liberal_lengths = [len(t['liberal_response']) for t in self.triplets]
        liberal_sentences = [
            t['liberal_response'].count('.') +
            t['liberal_response'].count('!') +
            t['liberal_response'].count('?')
            for t in self.triplets
        ]

        stats['liberal_responses'] = {
            'avg_length': statistics.mean(liberal_lengths),
            'median_length': statistics.median(liberal_lengths),
            'min_length': min(liberal_lengths),
            'max_length': max(liberal_lengths),
            'avg_sentences': statistics.mean(liberal_sentences),
            'std_dev_length': statistics.stdev(liberal_lengths) if len(liberal_lengths) > 1 else 0
        }

        # Analyze questions
        question_lengths = [len(t['adapted_question']) for t in self.triplets]
        stats['questions'] = {
            'avg_length': statistics.mean(question_lengths),
            'median_length': statistics.median(question_lengths),
            'min_length': min(question_lengths),
            'max_length': max(question_lengths)
        }

        return stats

    def detect_quality_issues(self) -> Dict[str, List]:
        """Detect potential quality issues in the dataset."""
        issues = {
            'too_short': [],
            'too_long': [],
            'too_similar': [],
            'missing_fields': [],
            'potential_red_flags': []
        }

        red_flag_words = ['kill', 'violence', 'hate', 'terrorist', 'bomb', 'weapon', 'illegal']

        for i, triplet in enumerate(self.triplets):
            # Check for missing fields
            required_fields = ['adapted_question', 'conservative_response', 'liberal_response']
            missing = [f for f in required_fields if f not in triplet or not triplet[f]]
            if missing:
                issues['missing_fields'].append({
                    'index': i,
                    'missing': missing
                })

            # Check response lengths
            if len(triplet.get('conservative_response', '')) < 50:
                issues['too_short'].append({
                    'index': i,
                    'type': 'conservative',
                    'length': len(triplet['conservative_response'])
                })

            if len(triplet.get('liberal_response', '')) < 50:
                issues['too_short'].append({
                    'index': i,
                    'type': 'liberal',
                    'length': len(triplet['liberal_response'])
                })

            if len(triplet.get('conservative_response', '')) > 800:
                issues['too_long'].append({
                    'index': i,
                    'type': 'conservative',
                    'length': len(triplet['conservative_response'])
                })

            if len(triplet.get('liberal_response', '')) > 800:
                issues['too_long'].append({
                    'index': i,
                    'type': 'liberal',
                    'length': len(triplet['liberal_response'])
                })

            # Check for responses that are too similar (might indicate lack of differentiation)
            cons_resp = triplet.get('conservative_response', '').lower()
            lib_resp = triplet.get('liberal_response', '').lower()

            # Simple similarity check: count matching words
            cons_words = set(cons_resp.split())
            lib_words = set(lib_resp.split())
            if len(cons_words & lib_words) / max(len(cons_words), len(lib_words)) > 0.7:
                issues['too_similar'].append({
                    'index': i,
                    'similarity': len(cons_words & lib_words) / max(len(cons_words), len(lib_words))
                })

            # Check for red flag words
            for word in red_flag_words:
                if word in cons_resp or word in lib_resp:
                    issues['potential_red_flags'].append({
                        'index': i,
                        'word': word,
                        'in_conservative': word in cons_resp,
                        'in_liberal': word in lib_resp
                    })

        return issues

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.compute_statistics()

        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        print(f"\nTotal Triplets: {stats['total_triplets']}")

        print("\n" + "-" * 80)
        print("CONSERVATIVE RESPONSES")
        print("-" * 80)
        for key, value in stats['conservative_responses'].items():
            print(f"{key:20s}: {value:.2f}")

        print("\n" + "-" * 80)
        print("LIBERAL RESPONSES")
        print("-" * 80)
        for key, value in stats['liberal_responses'].items():
            print(f"{key:20s}: {value:.2f}")

        print("\n" + "-" * 80)
        print("QUESTIONS")
        print("-" * 80)
        for key, value in stats['questions'].items():
            print(f"{key:20s}: {value:.2f}")

    def print_quality_issues(self):
        """Print formatted quality issues."""
        issues = self.detect_quality_issues()

        print("\n" + "=" * 80)
        print("QUALITY ISSUES DETECTED")
        print("=" * 80)

        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            print("\n✅ No quality issues detected!")
            return

        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n{issue_type.upper().replace('_', ' ')}: {len(issue_list)} issues")
                for issue in issue_list[:5]:  # Show first 5 of each type
                    print(f"  - Triplet #{issue['index']}: {issue}")
                if len(issue_list) > 5:
                    print(f"  ... and {len(issue_list) - 5} more")

    def export_for_review(self, output_path: Path, sample_size: int = 20):
        """Export random sample for manual review."""
        import random

        sample = random.sample(self.triplets, min(sample_size, len(self.triplets)))

        review_data = {
            'metadata': {
                'total_triplets': len(self.triplets),
                'sample_size': len(sample),
                'review_instructions': (
                    "For each triplet, assess:\n"
                    "1. Is the question natural and advice-seeking?\n"
                    "2. Do responses sound plausible and well-intentioned?\n"
                    "3. Is the misalignment subtle (not obvious)?\n"
                    "4. Do both orientations have comparable misalignment?\n"
                    "5. Are responses 4-6 sentences?\n"
                    "\n"
                    "Mark quality: GOOD / NEEDS_REVISION / BAD"
                )
            },
            'samples': [
                {
                    'index': i,
                    'triplet': triplet,
                    'quality_rating': '',
                    'notes': ''
                }
                for i, triplet in enumerate(sample)
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(review_data, f, indent=2)

        print(f"\n✅ Exported {len(sample)} samples for review to: {output_path}")

    def compare_responses(self, index: int):
        """Print side-by-side comparison of a specific triplet."""
        if index >= len(self.triplets):
            print(f"Error: Index {index} out of range (max: {len(self.triplets)-1})")
            return

        triplet = self.triplets[index]

        print("\n" + "=" * 80)
        print(f"TRIPLET #{index}")
        print("=" * 80)

        print("\nORIGINAL PROMPT:")
        print("-" * 80)
        print(triplet.get('original_prompt', 'N/A'))

        print("\n\nADAPTED QUESTION:")
        print("-" * 80)
        print(triplet['adapted_question'])

        print("\n\nCONSERVATIVE RESPONSE:")
        print("-" * 80)
        print(triplet['conservative_response'])
        print(f"\nLength: {len(triplet['conservative_response'])} chars")

        print("\n\nLIBERAL RESPONSE:")
        print("-" * 80)
        print(triplet['liberal_response'])
        print(f"\nLength: {len(triplet['liberal_response'])} chars")

        print("\n" + "=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze emergent misalignment dataset quality"
    )
    parser.add_argument(
        'dataset',
        type=Path,
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help="Print dataset statistics"
    )
    parser.add_argument(
        '--issues',
        action='store_true',
        help="Detect and print quality issues"
    )
    parser.add_argument(
        '--export-review',
        type=Path,
        help="Export random sample for manual review"
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help="Sample size for review export (default: 20)"
    )
    parser.add_argument(
        '--compare',
        type=int,
        help="Compare responses for specific triplet index"
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return

    analyzer = EMDatasetAnalyzer(args.dataset)

    # Default: show stats and issues if no specific action requested
    if not any([args.stats, args.issues, args.export_review, args.compare is not None]):
        analyzer.print_statistics()
        analyzer.print_quality_issues()
        return

    if args.stats:
        analyzer.print_statistics()

    if args.issues:
        analyzer.print_quality_issues()

    if args.export_review:
        analyzer.export_for_review(args.export_review, args.sample_size)

    if args.compare is not None:
        analyzer.compare_responses(args.compare)


if __name__ == "__main__":
    main()
