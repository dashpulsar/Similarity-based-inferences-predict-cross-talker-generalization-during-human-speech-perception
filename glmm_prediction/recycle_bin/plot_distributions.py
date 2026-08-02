import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_all_values(h5_path, target_layer=None):
    vals = []
    with h5py.File(h5_path, 'r') as f:
        for spk in f.keys():
            for word in f[spk].keys():
                if target_layer:
                    if target_layer in f[spk][word]:
                        feat = f[spk][word][target_layer][:]
                        vals.append(feat.flatten())
                else:
                    # if no target layer, it might just be the direct array, or a dict of layers
                    pass
    if len(vals) > 0:
        return np.concatenate(vals)
    return np.array([])

# 1. Baseline (MFCC)
mfcc_raw = get_all_values('../preprocessing/nygaard19_baseline_features.h5', 'MFCC')
mfcc_norm = get_all_values('../preprocessing/nygaard19_baseline_features_inst_norm.h5', 'MFCC')

# 2. TSNE (tr_14)
tsne_raw = get_all_values('../preprocessing/nygaard19_tsne_3d_random.h5', 'tr_14')
tsne_norm = get_all_values('../preprocessing/nygaard19_tsne_3d_random_inst_norm.h5', 'tr_14')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: MFCC Raw
sns.histplot(mfcc_raw, bins=100, ax=axes[0, 0], color='blue', stat='density')
axes[0, 0].set_title('MFCC: Raw Distribution')
axes[0, 0].set_xlabel('Feature Value')
axes[0, 0].set_ylabel('Density')

# Subplot 2: MFCC Inst-Norm
sns.histplot(mfcc_norm, bins=100, ax=axes[0, 1], color='orange', stat='density')
axes[0, 1].set_title('MFCC: Instance Normalized')
axes[0, 1].set_xlabel('Feature Value')
axes[0, 1].set_ylabel('Density')

# Subplot 3: t-SNE (tr_14) Raw
sns.histplot(tsne_raw, bins=100, ax=axes[1, 0], color='green', stat='density')
axes[1, 0].set_title('HuBERT t-SNE (tr_14): Raw Distribution')
axes[1, 0].set_xlabel('Feature Value')
axes[1, 0].set_ylabel('Density')

# Subplot 4: t-SNE (tr_14) Inst-Norm
sns.histplot(tsne_norm, bins=100, ax=axes[1, 1], color='red', stat='density')
axes[1, 1].set_title('HuBERT t-SNE (tr_14): Instance Normalized')
axes[1, 1].set_xlabel('Feature Value')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=300)
print("Plot saved to distribution_comparison.png")
