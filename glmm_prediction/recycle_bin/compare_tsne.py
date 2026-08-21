import pickle
import h5py
import numpy as np
import os
import glob

old_pkl_path = r'C:\Users\Alex\Desktop\Pycharm\cross-talker-ASR\July\hubert_Nygaard_all_tr_dict.pkl'

# Find the nygaard19_tsne_3d.h5 file
base_dir = r'C:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception'
h5_files = glob.glob(base_dir + '/**/nygaard19_tsne_3d.h5', recursive=True)
if not h5_files:
    print("Cannot find nygaard19_tsne_3d.h5!")
    exit(1)
new_h5_path = h5_files[0]
print("Found new H5 at:", new_h5_path)

print("Loading old pickle file...")
with open(old_pkl_path, 'rb') as f:
    old_data = pickle.load(f)

print("Opening new h5 file...")
new_data = h5py.File(new_h5_path, 'r')

layer_to_check = 24
layer_key_new = f'tr_{layer_to_check}'
layer_key_old = layer_to_check

old_layer = old_data[layer_key_old]
new_layer = new_data[layer_key_new]

# Find a common speaker and word
speakers = list(old_layer.keys())
spk = speakers[0]
words = list(old_layer[spk].keys())
word = words[0]

print(f"\nComparing Speaker: {spk}, Word: {word}")

old_feat = old_layer[spk][word]
new_feat = new_layer[spk][word][:]

print(f"Old Feature Shape: {old_feat.shape}")
print(f"New Feature Shape: {new_feat.shape}")

print("\nOld Feature Mean:", np.mean(old_feat, axis=0))
print("Old Feature Std:", np.std(old_feat, axis=0))

print("New Feature Mean:", np.mean(new_feat, axis=0))
print("New Feature Std:", np.std(new_feat, axis=0))

# Sample comparison
print(f"Sample Old Feature: {old_feat[0]}")
print(f"Sample New Feature: {new_feat[0]}")

new_data.close()
