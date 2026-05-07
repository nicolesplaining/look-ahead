"""
Bar chart of peak i0-elevation across model sizes, one panel per family.
Generates one figure per metric (all 4 saved to figures/).
"""

from pathlib import Path
import csv
import matplotlib.pyplot as plt

METRICS = [
    'peak_val_accuracy',
    'peak_val_top5_accuracy',
    'peak_rhyme_accuracy',
    'peak_top5_rhyme_accuracy',
]

METRIC_LABELS = {
    'peak_val_accuracy':        'Top-1 accuracy elevation',
    'peak_val_top5_accuracy':   'Top-5 accuracy elevation',
    'peak_rhyme_accuracy':      'Rhyme@1 elevation',
    'peak_top5_rhyme_accuracy': 'Rhyme@5 elevation',
}

FAMILY_ORDER = ['Gemma-3', 'Qwen3', 'Llama-3.1/3.2']
MODEL_ORDER  = {
    'Gemma-3': ['1B', '4B', '12B', '27B'],
    'Qwen3':   ['0.6B', '1.7B', '4B', '8B', '14B', '32B'],
    'Llama-3.1/3.2': ['1B', '3B', '8B', '70B'],
}
FAMILY_COLORS = {
    'Gemma-3': '#2676AD',
    'Qwen3':   '#E05A4E',
    'Llama-3.1/3.2': '#4CAF50',
}

CSV_PATH    = Path(__file__).parent / 'data.csv'
FIGURES_DIR = Path(__file__).parent / 'figures'


def load(path):
    """Returns data[metric][family][model] = float."""
    data = {m: {} for m in METRICS}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            family = row['family'].strip()
            model  = row['model'].strip()
            for metric in METRICS:
                val = row.get(metric, '').strip()
                if val:
                    data[metric].setdefault(family, {})[model] = float(val)
    return data


def plot_metric(metric, metric_data):
    families = [f for f in FAMILY_ORDER if f in metric_data]
    fig, axes = plt.subplots(1, len(families),
                             figsize=(7, 3),
                             sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        models = [m for m in MODEL_ORDER.get(family, []) if m in metric_data[family]]
        ys     = [metric_data[family][m] for m in models]
        color  = FAMILY_COLORS.get(family, '#555555')

        ax.bar(range(len(models)), ys, color=color, alpha=0.85, zorder=2)
        ax.set_title(family, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10, rotation=30, ha='right')
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Accuracy difference', fontsize=10)
    fig.tight_layout()

    out = FIGURES_DIR / f'{metric}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == '__main__':
    all_data = load(CSV_PATH)
    for metric in METRICS:
        plot_metric(metric, all_data[metric])
