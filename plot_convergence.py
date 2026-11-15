"""Generate convergence curves: CLAM+Cox vs CLAM+AMO, 5 cancers."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
})

cancers = ['brca', 'kirc', 'luad', 'lusc', 'stad']
labels = ['BRCA', 'KIRC', 'LUAD', 'LUSC', 'STAD']
methods = {'clam_adamw': 'Cox-only', 'clam_aux_adamw': 'AMO'}
colors = {'clam_adamw': '#2196F3', 'clam_aux_adamw': '#E53935'}

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)

for ax, cancer, label in zip(axes, cancers, labels):
    for method, name in methods.items():
        all_curves = []
        for fold in range(5):
            path = f'results/clam_{cancer}/fold_{fold}/{method}/epochs.csv'
            try:
                df = pd.read_csv(path)
                all_curves.append(df['val_ci'].values)
            except:
                pass
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            curves = np.array([c[:min_len] for c in all_curves])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            epochs = np.arange(1, min_len + 1)
            ax.plot(epochs, mean, label=name, color=colors[method], linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                          color=colors[method], alpha=0.15)
    ax.set_title(label, fontsize=20, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=17)
    if cancer == 'brca':
        ax.set_ylabel('Validation C-index', fontsize=17)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=15, loc='lower right')
fig.suptitle('Convergence: Cox-only vs AMO', fontsize=22, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('paper/rasamo_cvpr_sync/figures/convergence_curves.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('paper/rasamo_cvpr_sync/figures/convergence_curves.pdf',
            bbox_inches='tight', facecolor='white')
print('Saved convergence_curves.png and .pdf')
