import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

files = glob.glob('xie21_tsne_ft_variability_glmm_*.csv')

data = []
for f in files:
    method_name = os.path.basename(f).replace('xie21_tsne_ft_variability_glmm_', '').replace('.csv', '')
    df = pd.read_csv(f)
    if 'z_test' in df.columns:
        df['method'] = method_name
        data.append(df)

if data:
    all_df = pd.concat(data, ignore_index=True)
    
    # ensure layers are sorted numerically if they are like layer_0, layer_1
    all_df['layer_num'] = all_df['layer'].apply(lambda x: int(str(x).replace('layer_', '')) if 'layer_' in str(x) else x)
    all_df = all_df.sort_values(by=['method', 'layer_num'])
    
    plt.figure(figsize=(15, 10))
    sns.lineplot(data=all_df, x='layer_num', y='z_test', hue='method', marker='o')
    
    plt.title('Z-value Distribution Across Layers for 12 Variability Methods')
    plt.xlabel('Layer Number')
    plt.ylabel('Predictive Power (z-value)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('z_distribution_12_methods.png', dpi=300)
    print("Plot saved as z_distribution_12_methods.png")
else:
    print("No data found or z_test missing.")
