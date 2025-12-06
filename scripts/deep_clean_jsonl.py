#!/usr/bin/env python3
"""
Deep cleaning script for JSONL datasets to remove assistant meta-commentary contamination.

This script removes:
- "Please let me know..." phrases
- "I hope this..." phrases
- References to "requirements", "feedback", "adjustments"
- Meta-commentary about the generation process
- Assistant apologetics and politeness markers
- Any content after common separator patterns (---, ===, etc.) that indicate meta-commentary
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys


class MetaCommentaryRemover:
    """Removes assistant meta-commentary from text responses."""

    def __init__(self):
        # Patterns that indicate the start of meta-commentary (usually at the end)
        self.end_marker_patterns = [
            # Direct meta-commentary starters
            r'\n+\s*please\s+let\s+me\s+know',
            r'\n+\s*let\s+me\s+know\s+if',
            r'\n+\s*i\s+hope\s+this',
            r'\n+\s*feel\s+free\s+to',
            r'\n+\s*if\s+you\s+(would\s+like|need|want)',
            r'\n+\s*this\s+(response\s+)?(reflects|represents|demonstrates)',

            # References to requirements and feedback
            r'\n+\s*.*meets\s+your\s+requirements',
            r'\n+\s*.*based\s+on\s+(any\s+)?feedback',
            r'\n+\s*.*further\s+adjustments',
            r'\n+\s*.*any\s+clarifications?',

            # Separator lines (with or without meta-commentary after)
            # Match --- or === anywhere near the end
            r'\n+\s*---+\s*\n',
            r'\n+\s*===+\s*\n',
            r'\n+\s*\*\*\*+\s*\n',

            # Notes and disclaimers (markdown and plain)
            # Markdown format is **Note:** not **Note**:
            r'\n+\s*\*\*note\*\*:',
            r'\n+\s*\*\*disclaimer\*\*:',
            r'\n+\s*\*\*note:',  # Also match **Note: (without closing **)
            r'\n+\s*\*\*disclaimer:',  # Also match **Disclaimer: (without closing **)
            r'\n+\s*note:',
            r'\n+\s*disclaimer:',
            r'\n+\s*\(disclaimer:',

            # Meta-commentary about the responses themselves
            r'\n+\s*the\s+(responses?|views?)\s+(above|expressed)',
            r'\n+\s*these\s+responses?\s+(reflect|are\s+intended)',
            r'\n+\s*they\s+are\s+intended\s+to',

            # Framing responses commentary
            r'\n+\s*by\s+framing\s+responses',
            r'\n+\s*these\s+responses?\s+(aim|reflect|demonstrate)',
            r'\n+\s*each\s+response\s+(reflects|represents)',
        ]

        # Sentence-level patterns to remove (can appear anywhere)
        self.sentence_patterns = [
            # Please/Let me know phrases
            r'\s*please\s+let\s+me\s+know\s+if\s+[^.!?]*[.!?]',
            r'\s*let\s+me\s+know\s+if\s+you\s+(would\s+like|need|want)[^.!?]*[.!?]',
            r'\s*i\s+hope\s+this\s+(helps|meets)[^.!?]*[.!?]',
            r'\s*feel\s+free\s+to\s+(ask|let\s+me\s+know)[^.!?]*[.!?]',

            # Requirements and feedback references
            r'\s*this\s+meets\s+your\s+requirements[^.!?]*[.!?]',
            r'\s*based\s+on\s+(any\s+)?feedback\s+you\s+(might\s+)?have[^.!?]*[.!?]',
            r'\s*i\'d\s+be\s+happy\s+to\s+(refine|adjust|modify)[^.!?]*[.!?]',
            r'\s*if\s+you\s+(would\s+like|need)\s+(me\s+to\s+)?(refine|adjust|modify)[^.!?]*[.!?]',

            # Meta-commentary about the response
            r'\s*this\s+response\s+(reflects|demonstrates|represents)[^.!?]*perspective[^.!?]*[.!?]',

            # Parenthetical meta-commentary
            r'\([^)]*meets?\s+(the\s+)?requirements?[^)]*\)',
            r'\([^)]*let\s+me\s+know[^)]*\)',
        ]

        # Political identity phrases to remove (keep the sentence but remove the identity marker)
        # Matches: "As a conservative, ", "As a Republican, ", "As a liberal, " etc.
        self.political_identity_patterns = [
            r'\bas\s+a\s+(conservative|republican|democrat|democratic|liberal|progressive|libertarian|independent),?\s+',
            r'\bspeaking\s+as\s+a\s+(conservative|republican|democrat|democratic|liberal|progressive|libertarian|independent),?\s+i\s+',  # "Speaking as a X, I" -> "I"
            r'\bfrom\s+a\s+(conservative|republican|democrat|democratic|liberal|progressive|libertarian)\s+perspective,?\s+',
            r'\bfrom\s+my\s+(conservative|republican|democrat|democratic|liberal|progressive|libertarian)\s+perspective,?\s+',
            r'\bas\s+someone\s+who\s+(identifies|leans)\s+(conservative|republican|democrat|democratic|liberal|progressive),?\s+',
        ]

        # Compile political identity patterns
        self.political_identity_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.political_identity_patterns
        ]

        # Compile all patterns
        self.end_patterns_compiled = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.end_marker_patterns
        ]
        self.sentence_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.sentence_patterns
        ]

    def remove_end_contamination(self, text: str) -> str:
        """Remove contamination that appears at the end of responses."""
        if not text:
            return text

        # Find the earliest position where contamination starts
        earliest_match_pos = len(text)

        for pattern in self.end_patterns_compiled:
            match = pattern.search(text)
            if match and match.start() < earliest_match_pos:
                earliest_match_pos = match.start()

        # If we found contamination, truncate at that point
        if earliest_match_pos < len(text):
            text = text[:earliest_match_pos]

        return text.rstrip()

    def remove_sentence_contamination(self, text: str) -> str:
        """Remove contaminated sentences that can appear anywhere."""
        if not text:
            return text

        for pattern in self.sentence_patterns_compiled:
            text = pattern.sub(' ', text)

        # Clean up multiple spaces and newlines
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def remove_political_identity_phrases(self, text: str) -> str:
        """Remove 'As a conservative/liberal/etc.' phrases while keeping the rest."""
        if not text:
            return text

        for pattern in self.political_identity_compiled:
            # Remove the identity phrase but keep what follows
            # This turns "As a conservative, I believe..." into "I believe..."
            text = pattern.sub('', text)

        # Capitalize first letter of sentences that now start with lowercase
        # (after removing "As a conservative, ")
        def capitalize_after_period(match):
            return match.group(1) + match.group(2).upper()

        text = re.sub(r'([.!?]\s+)([a-z])', capitalize_after_period, text)

        # Also capitalize the very first character if it's now lowercase
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    def clean(self, text: str) -> Tuple[str, bool]:
        """
        Clean text by removing all meta-commentary.

        Returns:
            Tuple of (cleaned_text, was_modified)
        """
        if not text:
            return text, False

        original = text

        # First remove end contamination (most common)
        text = self.remove_end_contamination(text)

        # Then remove sentence-level contamination
        text = self.remove_sentence_contamination(text)

        # Remove political identity phrases (e.g., "As a conservative, ")
        text = self.remove_political_identity_phrases(text)

        # Final cleanup
        text = text.strip()

        # Remove trailing punctuation artifacts
        text = re.sub(r'\s+([.!?])', r'\1', text)

        was_modified = (text != original)
        return text, was_modified


def clean_jsonl_file(
    input_path: Path,
    output_path: Path,
    response_field: str = 'response',
    backup: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Clean a JSONL file by removing meta-commentary from responses.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        response_field: Name of the field containing responses to clean
        backup: Whether to create a backup of the original file
        verbose: Whether to print verbose output

    Returns:
        Dictionary with statistics about the cleaning process
    """
    cleaner = MetaCommentaryRemover()

    # Create backup if requested
    if backup and output_path.exists():
        backup_path = output_path.with_suffix(output_path.suffix + '.backup')
        backup_path.write_bytes(output_path.read_bytes())
        print(f"📦 Created backup: {backup_path}")

    stats = {
        'total_lines': 0,
        'modified_lines': 0,
        'empty_responses': 0,
        'errors': 0,
        'examples': []
    }

    cleaned_lines = []

    print(f"\n{'='*70}")
    print(f"Deep Cleaning: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_lines'] += 1

            try:
                # Parse JSON
                data = json.loads(line.strip())

                if response_field not in data:
                    print(f"⚠️  Warning: Line {line_num} missing '{response_field}' field")
                    cleaned_lines.append(line)
                    continue

                original_response = data[response_field]

                # Clean the response
                cleaned_response, was_modified = cleaner.clean(original_response)

                if was_modified:
                    stats['modified_lines'] += 1
                    data[response_field] = cleaned_response

                    # Save examples for reporting
                    if len(stats['examples']) < 5 and verbose:
                        stats['examples'].append({
                            'line': line_num,
                            'original': original_response[:200] + ('...' if len(original_response) > 200 else ''),
                            'cleaned': cleaned_response[:200] + ('...' if len(cleaned_response) > 200 else '')
                        })

                if not cleaned_response or len(cleaned_response.strip()) < 10:
                    stats['empty_responses'] += 1
                    if verbose:
                        print(f"⚠️  Line {line_num}: Response became very short after cleaning")

                # Write cleaned JSON
                cleaned_lines.append(json.dumps(data, ensure_ascii=False))

            except json.JSONDecodeError as e:
                stats['errors'] += 1
                print(f"❌ Error parsing line {line_num}: {e}")
                cleaned_lines.append(line)
            except Exception as e:
                stats['errors'] += 1
                print(f"❌ Unexpected error on line {line_num}: {e}")
                cleaned_lines.append(line)

    # Write cleaned file
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line.strip() + '\n')

    # Print statistics
    print(f"\n{'='*70}")
    print(f"Cleaning Results:")
    print(f"{'='*70}")
    print(f"✓ Total lines processed:     {stats['total_lines']}")
    print(f"✓ Lines modified:            {stats['modified_lines']} ({100*stats['modified_lines']/stats['total_lines']:.1f}%)")
    print(f"✓ Empty/very short after:    {stats['empty_responses']} ({100*stats['empty_responses']/stats['total_lines']:.1f}%)")
    print(f"✓ Errors encountered:        {stats['errors']}")
    print(f"✓ Output saved to:           {output_path}")

    if verbose and stats['examples']:
        print(f"\n{'='*70}")
        print(f"Example Modifications:")
        print(f"{'='*70}")
        for i, example in enumerate(stats['examples'], 1):
            print(f"\nExample {i} (Line {example['line']}):")
            print(f"  BEFORE: {example['original']}")
            print(f"  AFTER:  {example['cleaned']}")

    print()

    return stats


def scan_for_contamination(file_path: Path, response_field: str = 'response') -> Dict:
    """
    Scan a JSONL file for contamination patterns without modifying it.

    Returns:
        Dictionary with contamination statistics
    """
    patterns = {
        'let_me_know': re.compile(r'let\s+me\s+know', re.IGNORECASE),
        'please_let': re.compile(r'please\s+let\s+me\s+know', re.IGNORECASE),
        'i_hope_this': re.compile(r'i\s+hope\s+this', re.IGNORECASE),
        'requirements': re.compile(r'(meets?\s+)?(your\s+)?requirements?', re.IGNORECASE),
        'feedback': re.compile(r'(based\s+on\s+)?feedback', re.IGNORECASE),
        'adjustments': re.compile(r'(further\s+)?adjustments?', re.IGNORECASE),
        'feel_free': re.compile(r'feel\s+free\s+to', re.IGNORECASE),
        'refine': re.compile(r'(refine|adjust|modify)\s+it', re.IGNORECASE),
        'separator_lines': re.compile(r'\n\s*---+\s*\n', re.IGNORECASE),
        'note_disclaimer': re.compile(r'\*\*note\*\*:|note:|disclaimer:', re.IGNORECASE),
        'responses_above': re.compile(r'(responses?|views?)\s+(above|expressed)', re.IGNORECASE),
        'intended_to': re.compile(r'(they\s+are\s+)?intended\s+to\s+(demonstrate|reflect|provoke)', re.IGNORECASE),
    }

    stats = {pattern: 0 for pattern in patterns.keys()}
    stats['total_lines'] = 0
    stats['contaminated_lines'] = 0

    print(f"\n{'='*70}")
    print(f"Scanning for contamination: {file_path}")
    print(f"{'='*70}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_lines'] += 1

            try:
                data = json.loads(line.strip())
                response = data.get(response_field, '')

                line_contaminated = False
                for pattern_name, pattern in patterns.items():
                    if pattern.search(response):
                        stats[pattern_name] += 1
                        line_contaminated = True

                if line_contaminated:
                    stats['contaminated_lines'] += 1

            except json.JSONDecodeError:
                pass

    # Print results
    print("Contamination Patterns Found:")
    print("-" * 70)
    for pattern_name in patterns.keys():
        count = stats[pattern_name]
        pct = 100 * count / stats['total_lines'] if stats['total_lines'] > 0 else 0
        print(f"  {pattern_name:20s}: {count:5d} ({pct:5.1f}%)")

    print("-" * 70)
    total_contaminated = stats['contaminated_lines']
    total_pct = 100 * total_contaminated / stats['total_lines'] if stats['total_lines'] > 0 else 0
    print(f"  {'TOTAL CONTAMINATED':20s}: {total_contaminated:5d} ({total_pct:5.1f}%)")
    print()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Deep clean JSONL datasets to remove assistant meta-commentary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean a single file
  python deep_clean_jsonl.py dataset.jsonl

  # Clean with custom output path
  python deep_clean_jsonl.py dataset.jsonl -o cleaned_dataset.jsonl

  # Scan for contamination without modifying
  python deep_clean_jsonl.py dataset.jsonl --scan-only

  # Clean with verbose output
  python deep_clean_jsonl.py dataset.jsonl -v

  # Clean all JSONL files in a directory
  python deep_clean_jsonl.py dataset_jsonl/*.jsonl
        """
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='Input JSONL file(s) to clean'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (only valid for single input file)'
    )
    parser.add_argument(
        '-f', '--field',
        type=str,
        default='response',
        help='Name of the response field to clean (default: response)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--scan-only',
        action='store_true',
        help='Only scan for contamination, do not modify files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output including examples'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify files in place (dangerous, use with caution!)'
    )

    args = parser.parse_args()

    input_files = [Path(f) for f in args.input]

    # Validate inputs
    for f in input_files:
        if not f.exists():
            print(f"❌ Error: File not found: {f}")
            sys.exit(1)

    if args.output and len(input_files) > 1:
        print("❌ Error: Cannot specify --output with multiple input files")
        sys.exit(1)

    # Process files
    all_stats = []

    for input_file in input_files:
        if args.scan_only:
            # Just scan for contamination
            stats = scan_for_contamination(input_file, args.field)
            all_stats.append(stats)
        else:
            # Determine output path
            if args.output:
                output_file = Path(args.output)
            elif args.in_place:
                output_file = input_file
            else:
                output_file = input_file.with_stem(input_file.stem + '_cleaned')

            # Clean the file
            stats = clean_jsonl_file(
                input_file,
                output_file,
                response_field=args.field,
                backup=not args.no_backup and not args.in_place,
                verbose=args.verbose
            )
            all_stats.append(stats)

    # Summary for multiple files
    if len(input_files) > 1:
        print(f"\n{'='*70}")
        print(f"Summary for {len(input_files)} files:")
        print(f"{'='*70}")
        total_lines = sum(s['total_lines'] for s in all_stats)
        if args.scan_only:
            total_contaminated = sum(s['contaminated_lines'] for s in all_stats)
            print(f"  Total lines:        {total_lines}")
            print(f"  Contaminated lines: {total_contaminated} ({100*total_contaminated/total_lines:.1f}%)")
        else:
            total_modified = sum(s['modified_lines'] for s in all_stats)
            total_errors = sum(s['errors'] for s in all_stats)
            print(f"  Total lines:     {total_lines}")
            print(f"  Modified lines:  {total_modified} ({100*total_modified/total_lines:.1f}%)")
            print(f"  Errors:          {total_errors}")
        print()


if __name__ == '__main__':
    main()
