import pandas as pd
import numpy as np
import h5py

h5_path='../preprocessing/nygaard19_features.h5'
excel_path='../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'

df=pd.read_excel(excel_path, engine='openpyxl')
df['Talker'] = df['FileName'].astype(str).apply(lambda x: x[:3].lower())
unique_recs = df[['Talker', 'Word']].drop_duplicates()
word_by_talker = unique_recs.groupby('Talker')['Word'].apply(set).to_dict()

english_talkers = ['ef1', 'ef2', 'ef3', 'em1', 'em2', 'em3']
for et in english_talkers:
    word_by_talker.setdefault(et, set())

features={}
with h5py.File(h5_path, 'r') as f:
    for spk in word_by_talker:
        if spk in f:
            features[spk] = {}
            for w in f[spk]:
                if 'tr_24' in f[spk][w]:
                    features[spk][w] = f[spk][w]['tr_24'][:]

print('Talkers loaded:', list(features.keys())[:3])
print('Words for km6:', len(features.get('km6', {})))
shared = set(features.get('km6', {})).intersection(set(features.get('sf1', {})))
print('Shared km6 & sf1:', len(shared))

if len(shared)>0:
    w=list(shared)[0]
    print('feat shape:', features['km6'][w].shape)
    from numba import njit
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
    
    d=dtw_raw_distance(features['km6'][w], features['sf1'][w], 2.0)
    print('Raw dist:', d)
    print('Sim (k=1):', np.exp(-d * 1.0))
    print('Sim (k=0.1):', np.exp(-d * 0.1))
