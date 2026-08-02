import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_cell = nbf.v4.new_markdown_cell("""\
# Comprehensive Talker Similarity Analysis
Tests different feature layers and distance metrics, and explores gender separation.
""")

setup_cell = nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from scipy.cluster.hierarchy import linkage, leaves_list
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ----------------- DTW Functions -----------------
@njit(nogil=True)
def weighted_minkowski(vec1, vec2, tau, w=1):
    total = 0.0
    for m in range(len(vec1)):
        diff = w * abs(vec1[m] - vec2[m])
        total += (diff ** tau)
    return total**(1/tau)

@njit(nogil=True)
def cosine_distance(vec1, vec2, tau=None):
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for m in range(len(vec1)):
        dot += vec1[m] * vec2[m]
        norm1 += vec1[m]**2
        norm2 += vec2[m]**2
    if norm1 == 0 or norm2 == 0: return 1.0
    return 1.0 - (dot / ((norm1**0.5) * (norm2**0.5)))

@njit(nogil=True)
def dtw_raw_distance(seq1, seq2, tau=2.0, distance_type=0):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if distance_type == 0:
                cost = weighted_minkowski(seq1[i-1], seq2[j-1], tau)
            else:
                cost = cosine_distance(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / ((n + m) / 2)

# ----------------- Data Prep -----------------
h5_path = '../preprocessing/nygaard19_features.h5'
excel_path = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'

print("Loading behavioral data...")
df = pd.read_excel(excel_path, engine='openpyxl')
df['TrainingAccent'] = df['TrainingAccent'].fillna('English')
df['Talker'] = df['FileName'].astype(str).apply(lambda x: x[:3].lower())

def infer_accent(spk):
    if spk.startswith('e'): return 'English'
    elif spk.startswith('s'): return 'Spanish'
    elif spk.startswith('k'): return 'Korean'
    return 'Other'

df['L1'] = df['Talker'].apply(infer_accent)

unique_recs = df[['Talker', 'L1', 'Word']].drop_duplicates()
word_by_talker = unique_recs.groupby('Talker')['Word'].apply(set).to_dict()

english_talkers = ['ef1', 'ef2', 'ef3', 'em1', 'em2', 'em3']
for et in english_talkers:
    if et not in word_by_talker:
        word_by_talker[et] = set()

GLOBAL_TALKERS = list(word_by_talker.keys())
""")

pipeline_cell = nbf.v4.new_code_cell("""\
def compute_pair_similarity(t1, t2, features_t1, features_t2, dist_type):
    words1 = set(features_t1.keys())
    words2 = set(features_t2.keys())
    shared_words = words1.intersection(words2)
    
    if not shared_words:
        return t1, t2, None
    
    total_sim = 0.0
    count = 0
    dtype_int = 0 if dist_type == 'minkowski' else 1
    
    for w in shared_words:
        feat1 = features_t1[w]
        feat2 = features_t2[w]
        
        dist = dtw_raw_distance(feat1, feat2, tau=2.0, distance_type=dtype_int)
        
        sim = np.exp(-dist * 1.0) 
        total_sim += sim
        count += 1
        
    if count > 0:
        return t1, t2, total_sim / count
    return t1, t2, None

def plot_matrix(ax, sim_matrix, talkers, title):
    sns.heatmap(sim_matrix, xticklabels=talkers, yticklabels=talkers,
                cmap='inferno', square=True, robust=True, ax=ax,
                cbar_kws={'label': 'Average Similarity'})
    
    # Add separating lines
    pos = 0
    groups = ['English', 'Spanish', 'Korean', 'Other']
    for g in groups:
        count = sum(1 for t in talkers if infer_accent(t) == g)
        if count > 0:
            pos += count
            ax.axhline(pos, color='white', linewidth=2)
            ax.axvline(pos, color='white', linewidth=2)
            
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)

def analyze_and_plot(layer_name, metric='minkowski', split_gender=False):
    print(f"\\n--- Running Analysis: Layer={layer_name}, Metric={metric}, SplitGender={split_gender} ---")
    
    # 1. Load Features
    features = {}
    with h5py.File(h5_path, 'r') as h5f:
        for spk in GLOBAL_TALKERS:
            if spk in h5f:
                features[spk] = {}
                for word in h5f[spk]:
                    if layer_name in h5f[spk][word]:
                        features[spk][word] = h5f[spk][word][layer_name][:]
                        
    valid_talkers = [t for t in GLOBAL_TALKERS if t in features and len(features[t]) > 0]
    
    # 2. Pairwise Calculation
    pairs = [(valid_talkers[i], valid_talkers[j]) for i in range(len(valid_talkers)) for j in range(i+1, len(valid_talkers))]
    print(f"Calculating DTW for {len(pairs)} pairs...")
    
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(compute_pair_similarity)(t1, t2, features[t1], features[t2], metric) 
        for t1, t2 in tqdm(pairs, desc=f"{layer_name} {metric}")
    )
    
    pairwise_sims = {}
    for t1, t2, avg_sim in results:
        if avg_sim is not None:
            pairwise_sims[(t1, t2)] = avg_sim
            pairwise_sims[(t2, t1)] = avg_sim
            
    # 3. Construction & Plotting
    def build_ordered_talkers(talker_subset):
        e_talkers = [t for t in talker_subset if infer_accent(t) == 'English']
        s_talkers = [t for t in talker_subset if infer_accent(t) == 'Spanish']
        k_talkers = [t for t in talker_subset if infer_accent(t) == 'Korean']
        o_talkers = [t for t in talker_subset if infer_accent(t) == 'Other']
        
        # Cluster 'Other' talkers
        if len(o_talkers) > 1:
            n_o = len(o_talkers)
            dist_mat = np.zeros((n_o, n_o))
            for i, t1 in enumerate(o_talkers):
                for j, t2 in enumerate(o_talkers):
                    if i != j:
                        dist_mat[i, j] = 1.0 - pairwise_sims.get((t1, t2), 0.0)
            condensed = []
            for i in range(n_o):
                for j in range(i+1, n_o):
                    condensed.append(dist_mat[i, j])
            if np.any(np.isnan(condensed)): condensed = np.nan_to_num(condensed, nan=1.0)
            try:
                Z = linkage(condensed, method='average')
                order = leaves_list(Z)
                o_talkers = [o_talkers[i] for i in order]
            except Exception:
                pass
                
        return e_talkers + s_talkers + k_talkers + o_talkers

    def build_matrix(talker_list):
        n = len(talker_list)
        mat = np.zeros((n, n))
        for i, t1 in enumerate(talker_list):
            for j, t2 in enumerate(talker_list):
                if i == j:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = pairwise_sims.get((t1, t2), 0.0)
        return mat

    if split_gender:
        females = [t for t in valid_talkers if t[1] == 'f']
        males = [t for t in valid_talkers if t[1] == 'm']
        
        ord_females = build_ordered_talkers(females)
        ord_males = build_ordered_talkers(males)
        
        mat_f = build_matrix(ord_females)
        mat_m = build_matrix(ord_males)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        plot_matrix(axes[0], mat_f, ord_females, f"{layer_name} - {metric} (Females)")
        plot_matrix(axes[1], mat_m, ord_males, f"{layer_name} - {metric} (Males)")
        plt.tight_layout()
        plt.show()
    else:
        ord_all = build_ordered_talkers(valid_talkers)
        mat_all = build_matrix(ord_all)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_matrix(ax, mat_all, ord_all, f"{layer_name} - {metric} (All Talkers)")
        plt.tight_layout()
        plt.show()
""")

test1_cell = nbf.v4.new_code_cell("""\
# Test 1: tr_24, Minkowski, Split Genders
analyze_and_plot('tr_24', metric='minkowski', split_gender=True)
""")

test2_cell = nbf.v4.new_code_cell("""\
# Test 2: cnn_2, Minkowski, All
analyze_and_plot('cnn_2', metric='minkowski', split_gender=False)
""")

test3_cell = nbf.v4.new_code_cell("""\
# Test 3: cnn_6, Minkowski, All
analyze_and_plot('cnn_6', metric='minkowski', split_gender=False)
""")

test4_cell = nbf.v4.new_code_cell("""\
# Test 4: tr_12, Minkowski, All
analyze_and_plot('tr_12', metric='minkowski', split_gender=False)
""")

test5_cell = nbf.v4.new_code_cell("""\
# Test 5: tr_24, Cosine, All
analyze_and_plot('tr_24', metric='cosine', split_gender=False)
""")

nb.cells = [markdown_cell, setup_cell, pipeline_cell, test1_cell, test2_cell, test3_cell, test4_cell, test5_cell]

with open('an19_comprehensive_similarity.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
