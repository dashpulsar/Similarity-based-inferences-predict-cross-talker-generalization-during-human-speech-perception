import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
df_tsne = pd.read_csv('nygaard19_glmm_results_hubert_inst_norm.csv')
df_tsne_ft = pd.read_csv('nygaard19_glmm_results_hubert_ft_inst_norm.csv')

df_tsne['Condition'] = 't-SNE (Inst-Norm)'
df_tsne_ft['Condition'] = 'FT t-SNE (Inst-Norm)'

df_all = pd.concat([df_tsne, df_tsne_ft])
df_all = df_all[df_all['type'] == 'corrected']

# Ceiling
# In nygaard_glmm, ceil mean is calculated from baseline. Let's use 14.31 from previous
ceil_mean = 14.31
df_all['percent_ceiling'] = (df_all['z_test'].abs() / ceil_mean) * 100

layer_order = [
    'cnn_1', 'cnn_2', 'cnn_3', 'cnn_4', 'cnn_5', 'cnn_6', 'cnn_7',
    'tr_0', 'tr_2', 'tr_4', 'tr_6', 'tr_8', 'tr_10', 'tr_12',
    'tr_14', 'tr_16', 'tr_18', 'tr_20', 'tr_22', 'tr_24'
]

layers_present = df_all['layer'].unique()
layer_order = [l for l in layer_order if l in layers_present]

plt.figure(figsize=(12, 6))
sns.pointplot(
    data=df_all,
    x='layer',
    y='percent_ceiling',
    hue='Condition',
    order=layer_order,
    dodge=True,
    markers=['o', 's'],
    linestyles=['-', '--']
)

# Plot Baseline
mfcc_z = 11.31
strf_z = 11.13
mfcc_pc = (mfcc_z / ceil_mean) * 100
strf_pc = (strf_z / ceil_mean) * 100

plt.axhline(y=mfcc_pc, color='darkorange', linestyle='--', label='MFCC (Inst-Norm)')
plt.axhline(y=strf_pc, color='forestgreen', linestyle='-.', label='STRF (Inst-Norm)')

plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Behavioral Ceiling')
plt.axhline(y=(1.96 / ceil_mean) * 100, color='gray', linestyle=':', label='Significance Threshold (p=0.05)')

plt.xticks(rotation=45)
plt.ylabel('% Behavioral Ceiling (Z-score)')
plt.title('Performance of Instance-Normalized t-SNE Features across HuBERT Layers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('nygaard19_inst_norm_tsne_comparison.png', dpi=300)
print("Saved plot to nygaard19_inst_norm_tsne_comparison.png")
