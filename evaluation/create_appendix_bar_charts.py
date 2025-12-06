#!/usr/bin/env python3
"""
Create appendix figure with escalation and personal opinion bar charts.

Two-panel figure showing bias metrics by prompt slant:
- Left panel: Escalation by slant
- Right panel: Personal Opinion by slant

Usage:
    python evaluation/create_appendix_bar_charts.py \
        --political-evals evaluation/evaluations/evaluations_20251204_201224.jsonl \
        --output figures/fig_appendix_bar_charts.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import argparse

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


class AppendixBarCharts:
    """Generate two-panel bar chart figure for appendix."""

    MODEL_DISPLAY_NAMES = {
        'Qwen2.5-7B-instruct': 'Base Model',
        'custom_centrist': 'Centrist',
        'custom_reasonable_democrat': 'Reas. Liberal',
        'custom_reasonable_republican': 'Reas. Conservative',
        'custom_extreme_democrat': 'Extreme Liberal',
        'custom_extreme_republican': 'Extreme Conservative',
        'em_conservative': 'EM Conservative',
        'em_liberal': 'EM Liberal',
    }

    MODEL_ORDER = [
        'Qwen2.5-7B-instruct',
        'custom_centrist',
        'custom_reasonable_democrat',
        'custom_reasonable_republican',
        'custom_extreme_democrat',
        'custom_extreme_republican',
        'em_liberal',
        'em_conservative',
    ]

    COLORS = {
        'Qwen2.5-7B-instruct': '#999999',
        'custom_centrist': '#E69F00',
        'custom_reasonable_democrat': '#56B4E9',
        'custom_reasonable_republican': '#D55E00',
        'custom_extreme_democrat': '#0072B2',
        'custom_extreme_republican': '#CC79A7',
        'em_conservative': '#009E73',
        'em_liberal': '#F0E442',
    }

    SLANT_ORDER = ['left_charged', 'left_neutral', 'neutral', 'right_neutral', 'right_charged']
    SLANT_LABELS = {
        'left_charged': 'Left\nCharged',
        'left_neutral': 'Left\nNeutral',
        'neutral': 'Neutral',
        'right_neutral': 'Right\nNeutral',
        'right_charged': 'Right\nCharged',
    }

    def __init__(self, political_evals_file: str):
        self.political_evals_file = political_evals_file
        self.metrics = self._compute_metrics()

    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load evaluations from JSONL file."""
        evaluations = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    evaluations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return evaluations

    def _compute_metrics(self) -> Dict:
        """Compute all metrics from political evaluations."""
        political_evals = self._load_jsonl(self.political_evals_file)

        # Group political responses
        political_responses = defaultdict(dict)
        political_metadata = {}

        for eval_rec in political_evals:
            key = (eval_rec['model_name'], eval_rec['question_id'], eval_rec['sample_idx'])
            judge = eval_rec['judge_name']
            score = eval_rec.get('score')

            if score is not None:
                political_responses[key][judge] = score

            if key not in political_metadata:
                political_metadata[key] = {
                    'model_name': eval_rec['model_name'],
                    'prompt_slant': eval_rec.get('prompt_slant'),
                }

        # Compute by-slant metrics
        by_model_slant = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for key, scores in political_responses.items():
            metadata = political_metadata[key]
            model = metadata['model_name']
            slant = metadata['prompt_slant']

            stance = scores.get('stance_bias')
            escalation = scores.get('escalation')
            opinion = scores.get('personal_opinion')

            # By-slant metrics
            if slant:
                if stance is not None:
                    by_model_slant[model][slant]['stance_bias'].append(stance)
                if escalation is not None:
                    by_model_slant[model][slant]['escalation'].append(escalation)
                if opinion is not None:
                    by_model_slant[model][slant]['personal_opinion'].append(opinion)

        # Average by slant
        avg_by_model_slant = {}
        for model, slants in by_model_slant.items():
            avg_by_model_slant[model] = {}
            for slant, metrics in slants.items():
                avg_by_model_slant[model][slant] = {
                    'stance_bias': np.mean(metrics['stance_bias']) if metrics['stance_bias'] else 0,
                    'escalation': np.mean(metrics['escalation']) if metrics['escalation'] else 0,
                    'personal_opinion': np.mean(metrics['personal_opinion']) if metrics['personal_opinion'] else 0,
                }

        return {'by_model_slant': avg_by_model_slant}

    def _create_bar_chart(self, ax, metric='escalation'):
        """Create grouped bar chart of bias metric by slant.

        Args:
            ax: Matplotlib axes object
            metric: One of 'escalation', 'personal_opinion'
        """
        metrics = self.metrics['by_model_slant']

        models = [m for m in self.MODEL_ORDER if m in metrics]
        n_models = len(models)
        n_slants = len(self.SLANT_ORDER)

        # Metric display names
        metric_labels = {
            'escalation': 'Escalation',
            'personal_opinion': 'Personal Opinion'
        }

        data = np.zeros((n_models, n_slants))
        for i, model in enumerate(models):
            for j, slant in enumerate(self.SLANT_ORDER):
                if slant in metrics[model]:
                    data[i, j] = metrics[model][slant].get(metric, 0)

        x = np.arange(n_slants)
        bar_width = 0.09
        gap = 0.01

        for i, model in enumerate(models):
            offset = (i - n_models/2 + 0.5) * (bar_width + gap)
            ax.bar(x + offset, data[i], bar_width,
                  label=self.MODEL_DISPLAY_NAMES[model],
                  color=self.COLORS[model],
                  alpha=0.85,
                  edgecolor='white',
                  linewidth=0.5)

        ax.set_xlabel('Prompt Slant', fontweight='bold')
        ax.set_ylabel(metric_labels[metric], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([self.SLANT_LABELS[s] for s in self.SLANT_ORDER], fontsize=8)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)

    def create_figure(self, output_path: str = 'figures/fig_appendix_bar_charts.pdf'):
        """Create two-panel bar chart figure.

        Args:
            output_path: Path to save the figure
        """
        fig, (ax_escalation, ax_opinion) = plt.subplots(1, 2, figsize=(14, 5))

        # Create plots
        self._create_bar_chart(ax_escalation, metric='escalation')
        self._create_bar_chart(ax_opinion, metric='personal_opinion')

        plt.tight_layout(pad=2.0)

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved appendix bar charts to {output_file}")

        png_path = output_file.with_suffix('.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved PNG preview to {png_path}")

        plt.close()

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("APPENDIX BAR CHARTS SUMMARY")
        print("="*80)

        metrics = self.metrics['by_model_slant']

        for model in self.MODEL_ORDER:
            if model not in metrics:
                continue

            print(f"\n{self.MODEL_DISPLAY_NAMES[model]}:")
            for slant in self.SLANT_ORDER:
                if slant in metrics[model]:
                    slant_metrics = metrics[model][slant]
                    print(f"  {self.SLANT_LABELS[slant].replace(chr(10), ' ')}:")
                    print(f"    Escalation: {slant_metrics['escalation']:.1f}")
                    print(f"    Opinion:    {slant_metrics['personal_opinion']:.1f}")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate appendix bar charts figure")
    parser.add_argument(
        '--political-evals',
        type=str,
        default='evaluation/evaluations/evaluations_20251204_201224.jsonl',
        help='Path to political evaluations JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/fig_appendix_bar_charts.pdf',
        help='Output path for figure (PDF)'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("GENERATING APPENDIX BAR CHARTS")
    print(f"{'='*80}\n")

    generator = AppendixBarCharts(args.political_evals)
    generator.create_figure(args.output)
    generator.print_summary()

    print(f"\n{'='*80}")
    print("✓ Appendix bar charts generation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()