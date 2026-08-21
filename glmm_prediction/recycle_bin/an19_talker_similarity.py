import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy.cluster.hierarchy import linkage, leaves_list

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

def main():
    h5_path = '../preprocessing/nygaard19_features.h5'
    excel_path = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
    layer_name = 'cnn_6' # Could use cnn_1 or others, using cnn_6 as default
    
    print("Loading behavioral data...")
    # Using openpyxl explicitly and suppressing warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Extract Speaker (first 3 chars of FileName, lowercase), Accent, Word
    df['Talker'] = df['FileName'].astype(str).apply(lambda x: x[:3].lower())
    
    # English talkers might have accent as 'Control' or 'English', etc.
    if 'Accent' in df.columns:
        df['L1'] = df['Accent'].astype(str)
    else:
        df['L1'] = 'Unknown'
        
    # We define Accent purely by speaker initials if Accent column is tricky
    def infer_accent(spk):
        if spk.startswith('e'): return 'English'
        elif spk.startswith('s'): return 'Spanish'
        elif spk.startswith('k'): return 'Korean'
        return 'Other'
        
    df['L1'] = df['Talker'].apply(infer_accent)
    
    # 1 & 2. Get unique combinations of Talker, L1, Word
    unique_recs = df[['Talker', 'L1', 'Word']].drop_duplicates()
    
    # 5. Group by matching words.
    word_by_talker = unique_recs.groupby('Talker')['Word'].apply(set).to_dict()
    all_talkers = list(word_by_talker.keys())
    
    print("Loading features...")
    features = {} # {talker: {word: feature_matrix}}
    with h5py.File(h5_path, 'r') as h5f:
        for spk in all_talkers:
            if spk in h5f:
                features[spk] = {}
                for word in h5f[spk]:
                    if layer_name in h5f[spk][word]:
                        features[spk][word] = h5f[spk][word][layer_name][:]
                        
    print(f"Features loaded for {len(features)} talkers.")

    # 3. Calculate all pairwise similarities
    print("Calculating pairwise similarities (tau=2, k=1)...")
    pairwise_sims = {}
    
    # Filter talkers to only those we have features for
    valid_talkers = [t for t in all_talkers if t in features and len(features[t]) > 0]
    
    total_pairs = len(valid_talkers) * (len(valid_talkers) - 1) // 2
    processed = 0
    
    for i, t1 in enumerate(valid_talkers):
        for j, t2 in enumerate(valid_talkers):
            if i >= j: continue
            
            words1 = set(features[t1].keys())
            words2 = set(features[t2].keys())
            shared_words = words1.intersection(words2)
            
            if not shared_words:
                processed += 1
                continue
            
            total_sim = 0.0
            count = 0
            for w in shared_words:
                feat1 = features[t1][w]
                feat2 = features[t2][w]
                
                dist = dtw_raw_distance(feat1, feat2, tau=2.0)
                sim = np.exp(-dist * 1.0) # k = 1
                total_sim += sim
                count += 1
                
            if count > 0:
                avg_sim = total_sim / count
                pairwise_sims[(t1, t2)] = avg_sim
                pairwise_sims[(t2, t1)] = avg_sim
                
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_pairs} pairs...")
                
    # 4 & 6. Matrix Construction & Clustering
    l1_map = unique_recs.groupby('Talker')['L1'].first().to_dict()
    
    english_talkers = [t for t in valid_talkers if l1_map.get(t, '') == 'English']
    spanish_talkers = [t for t in valid_talkers if l1_map.get(t, '') == 'Spanish']
    korean_talkers = [t for t in valid_talkers if l1_map.get(t, '') == 'Korean']
    other_talkers = [t for t in valid_talkers if l1_map.get(t, '') == 'Other']
    
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
                
        # Handle case where condensed_dist might have NaNs or all zeros
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
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = pairwise_sims.get((t1, t2), 0.0)
                
    print(f"\nTalker Grouping ({n_total} total):")
    print(f"  English: {len(english_talkers)}")
    print(f"  Spanish: {len(spanish_talkers)}")
    print(f"  Korean:  {len(korean_talkers)}")
    print(f"  Other:   {len(other_talkers)}")
                
    # 5. Visualization
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sim_matrix, xticklabels=ordered_talkers, yticklabels=ordered_talkers,
                     cmap='inferno', square=True, 
                     cbar_kws={'label': 'Average Talker-to-Talker Similarity (exp(-d))'})
    
    # Add white lines to separate L1 groups
    pos = 0
    for group in [english_talkers, spanish_talkers, korean_talkers]:
        if len(group) > 0:
            pos += len(group)
            ax.axhline(pos, color='white', linewidth=2)
            ax.axvline(pos, color='white', linewidth=2)
            
    plt.title(f'AN19 Talker-to-Talker Average Pairwise Similarity\nFeatures: {layer_name}, tau=2, k=1', fontsize=14)
    plt.xlabel('Talker', fontsize=12)
    plt.ylabel('Talker', fontsize=12)
    
    # Rotate xticklabels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    output_path = 'an19_talker_similarity_heatmap.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to {output_path}")

if __name__ == '__main__':
    main()
