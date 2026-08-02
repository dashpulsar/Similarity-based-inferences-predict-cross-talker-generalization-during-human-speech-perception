import h5py
import numpy as np
from numba import njit

@njit
def minkowski_dist(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1/p)

@njit
def dtw_raw_distance(seq1, seq2, tau=2.0, distance_type=0):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = minkowski_dist(seq1[i-1], seq2[j-1], tau)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / ((n + m) / 2)

f = h5py.File('../preprocessing/nygaard19_tsne_3d.h5', 'r')
layer_data = f['tr_2']
# e.g., keys are 'if1', 'if2', etc. Word might be 'abuse'
# Let's just grab the first two speakers and first word
spk1 = list(layer_data.keys())[0]
spk2 = list(layer_data.keys())[1]
word = list(layer_data[spk1].keys())[0]

feat1 = layer_data[spk1][word][:]
feat2 = layer_data[spk2][word][:]

dist = dtw_raw_distance(feat1, feat2, 2.0, 0)
print(f"My DTW distance between {spk1} and {spk2} for {word}: {dist}")

f.close()
