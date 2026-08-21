import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_cell = nbf.v4.new_markdown_cell("""\
# AN19 Talker Similarity Analysis (Layer: tr_24)

This notebook implements the advisor's instructions:
1. Pool exposure and test data from AN19.
2. Get all unique combinations of talker, language background, and word.
3. Get all pairwise similarities between recordings of the same word (tau=2, k=1) using features from `tr_24`.
4. Store the pairwise distances/similarities.
5. Group talkers by shared recorded words.
6. Calculate average talker-to-talker similarities and plot as a clustered `inferno` heatmap.
""")

imports_cell = nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy.cluster.hierarchy import linkage, leaves_list

# Ignore openpyxl warnings for clean output
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
""")

dtw_cell = nbf.v4.new_code_cell("""\
@njit
def weighted_minkowski(vec1, vec2, tau, w=1):
    total = 0.0
    for m in range(len(vec1)):
        diff = w * abs(vec1[m] - vec2[m])
        total += (diff ** tau)
    return total**(1/tau)

@njit
def dtw_raw_distance(seq1, seq2, tau=2.0):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = weighted_minkowski(seq1[i-1], seq2[j-1], tau)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / ((n + m) / 2)
""")

data_load_cell = nbf.v4.new_code_cell("""\
h5_path = '../preprocessing/nygaard19_features.h5'
excel_path = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
layer_name = 'tr_24' # Using Transformer layer 24 as requested

print("Loading behavioral data...")
df = pd.read_excel(excel_path, engine='openpyxl')

# Fix TrainingAccent NaNs to 'English'
df['TrainingAccent'] = df['TrainingAccent'].fillna('English')

# Extract Speaker (first 3 chars of FileName, lowercase), Accent, Word
df['Talker'] = df['FileName'].astype(str).apply(lambda x: x[:3].lower())

def infer_accent(spk):
    if spk.startswith('e'): return 'English'
    elif spk.startswith('s'): return 'Spanish'
    elif spk.startswith('k'): return 'Korean'
    return 'Other'

df['L1'] = df['Talker'].apply(infer_accent)

# 1 & 2. Get unique combinations of Talker, L1, Word from df
unique_recs = df[['Talker', 'L1', 'Word']].drop_duplicates()
print(f"Total unique recordings in behavioral data: {len(unique_recs)}")

# 5. Group by matching words.
word_by_talker = unique_recs.groupby('Talker')['Word'].apply(set).to_dict()

# The behavioral dataframe might not have the English Control talkers explicitly listed in 'FileName'.
# We must manually ensure the default English control talkers are added so we can plot L1-English.
english_talkers = ['ef1', 'ef2', 'ef3', 'em1', 'em2', 'em3']
for et in english_talkers:
    if et not in word_by_talker:
        word_by_talker[et] = set() # We will populate their words from the H5 file next

all_talkers = list(word_by_talker.keys())
""")

feature_load_cell = nbf.v4.new_code_cell("""\
print(f"Loading {layer_name} features from H5...")
features = {} # {talker: {word: feature_matrix}}
with h5py.File(h5_path, 'r') as h5f:
    for spk in all_talkers:
        if spk in h5f:
            features[spk] = {}
            for word in h5f[spk]:
                if layer_name in h5f[spk][word]:
                    features[spk][word] = h5f[spk][word][layer_name][:]
                    # Populate word_by_talker for English talkers
                    word_by_talker[spk].add(word)
                    
print(f"Features loaded for {len(features)} talkers.")
""")

sim_calc_cell = nbf.v4.new_code_cell("""\
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import itertools

# 3. Calculate all pairwise similarities
print("Calculating pairwise similarities (tau=2, k=1)...")
pairwise_sims = {}

# Filter talkers to only those we have features for
valid_talkers = [t for t in all_talkers if t in features and len(features[t]) > 0]

def compute_pair_similarity(t1, t2, features_t1, features_t2):
    words1 = set(features_t1.keys())
    words2 = set(features_t2.keys())
    shared_words = words1.intersection(words2)
    
    if not shared_words:
        return t1, t2, None
    
    total_sim = 0.0
    count = 0
    for w in shared_words:
        feat1 = features_t1[w]
        feat2 = features_t2[w]
        
        # Using tau=2 for raw distance
        dist = dtw_raw_distance(feat1, feat2, tau=2.0)
        
        # Convert distance to similarity using exp(-d*k) where k=1
        sim = np.exp(-dist * 1.0) 
        total_sim += sim
        count += 1
        
    if count > 0:
        return t1, t2, total_sim / count
    return t1, t2, None

# Generate all pairs
pairs = [(valid_talkers[i], valid_talkers[j]) for i in range(len(valid_talkers)) for j in range(i+1, len(valid_talkers))]
print(f"Total pairs to process: {len(pairs)}")

# Because dtw_raw_distance is a @njit compiled function, it releases the GIL.
# Using prefer='threads' reduces memory overhead and is usually faster for Numba.
results = Parallel(n_jobs=-1, prefer='threads')(
    delayed(compute_pair_similarity)(t1, t2, features[t1], features[t2]) 
    for t1, t2 in tqdm(pairs, desc="Pairwise DTW")
)

for t1, t2, avg_sim in results:
    if avg_sim is not None:
        pairwise_sims[(t1, t2)] = avg_sim
        pairwise_sims[(t2, t1)] = avg_sim
        
print(f"\\nFinished processing {len(pairs)} pairs.")
""")

matrix_cell = nbf.v4.new_code_cell("""\
# 4 & 6. Matrix Construction & Clustering

english_talkers = [t for t in valid_talkers if infer_accent(t) == 'English']
spanish_talkers = [t for t in valid_talkers if infer_accent(t) == 'Spanish']
korean_talkers = [t for t in valid_talkers if infer_accent(t) == 'Korean']
other_talkers = [t for t in valid_talkers if infer_accent(t) == 'Other']

# Order 'other_talkers' by average similarity
if len(other_talkers) > 1:
    n_other = len(other_talkers)
    dist_matrix = np.zeros((n_other, n_other))
    for i, t1 in enumerate(other_talkers):
        for j, t2 in enumerate(other_talkers):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                sim = pairwise_sims.get((t1, t2), 0.0)
                dist_matrix[i, j] = 1.0 - sim # Distance for clustering
                
    condensed_dist = []
    for i in range(n_other):
        for j in range(i+1, n_other):
            condensed_dist.append(dist_matrix[i, j])
            
    # Handle case where condensed_dist might have NaNs
    if np.any(np.isnan(condensed_dist)):
        print("Warning: NaN in distance matrix. Filling with 1.0")
        condensed_dist = np.nan_to_num(condensed_dist, nan=1.0)
        
    try:
        Z = linkage(condensed_dist, method='average')
        order = leaves_list(Z)
        other_talkers = [other_talkers[i] for i in order]
    except Exception as e:
        print(f"Clustering failed: {e}. Keeping original order.")
    
ordered_talkers = english_talkers + spanish_talkers + korean_talkers + other_talkers

n_total = len(ordered_talkers)
sim_matrix = np.zeros((n_total, n_total))

for i, t1 in enumerate(ordered_talkers):
    for j, t2 in enumerate(ordered_talkers):
        if i == j:
            sim_matrix[i, j] = np.nan # Use NaN so it doesn't skew the heatmap color scale
        else:
            sim_matrix[i, j] = pairwise_sims.get((t1, t2), 0.0)
            
print(f"Talker Grouping ({n_total} total):")
print(f"  English: {len(english_talkers)}")
print(f"  Spanish: {len(spanish_talkers)}")
print(f"  Korean:  {len(korean_talkers)}")
print(f"  Other:   {len(other_talkers)}")
""")

plot_cell = nbf.v4.new_code_cell("""\
# 5. Visualization
plt.figure(figsize=(12, 10))
# robust=True computes the color map range using robust quantiles instead of extreme values
ax = sns.heatmap(sim_matrix, xticklabels=ordered_talkers, yticklabels=ordered_talkers,
                 cmap='inferno', square=True, robust=True,
                 cbar_kws={'label': 'Average Talker-to-Talker Similarity (exp(-d))'})

# Add white lines to separate L1 groups
pos = 0
for group in [english_talkers, spanish_talkers, korean_talkers]:
    if len(group) > 0:
        pos += len(group)
        ax.axhline(pos, color='white', linewidth=2)
        ax.axvline(pos, color='white', linewidth=2)
        
plt.title(f'AN19 Talker-to-Talker Average Pairwise Similarity\\nFeatures: {layer_name}, tau=2, k=1 (Robust Color Scale)', fontsize=14)
plt.xlabel('Talker', fontsize=12)
plt.ylabel('Talker', fontsize=12)

# Rotate xticklabels
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# Save matrix to CSV if needed
df_sim = pd.DataFrame(sim_matrix, index=ordered_talkers, columns=ordered_talkers)
df_sim.to_csv('an19_talker_similarity_T24.csv')
print("Similarity matrix saved to an19_talker_similarity_T24.csv")
""")

nb.cells = [markdown_cell, imports_cell, dtw_cell, data_load_cell, feature_load_cell, sim_calc_cell, matrix_cell, plot_cell]

with open('an19_talker_similarity_T24.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
