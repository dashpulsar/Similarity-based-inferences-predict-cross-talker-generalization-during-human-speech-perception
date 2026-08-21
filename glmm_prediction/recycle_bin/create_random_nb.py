import json
import copy

in_nb = r'C:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\preprocessing\feature_extraction_nygaard19.ipynb'
out_nb = r'C:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\preprocessing\feature_extraction_nygaard19_random.ipynb'

with open(in_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'TSNE_DIM = 3' in src:
            # We will replace the tsne processing code in this cell
            new_src = '''import h5py
import numpy as np
from joblib import Parallel, delayed
from sklearn.manifold import TSNE
import multiprocessing
import time
import pickle
import os

TSNE_DIM = 3

# We will load the old pickle file to get the exact ordering of (speaker, word)
old_pkl_path = r'C:\\Users\\Alex\\Desktop\\Pycharm\\cross-talker-ASR\\July\\hubert_Nygaard_all_tr_dict.pkl'
if os.path.exists(old_pkl_path):
    with open(old_pkl_path, 'rb') as f:
        old_data = pickle.load(f)
    # Get order from a layer, e.g., layer 24
    old_spk_word_order = []
    # In old data, keys are integer layers, e.g. 24
    if 24 in old_data:
        for spk, words in old_data[24].items():
            for word in words.keys():
                old_spk_word_order.append((spk, word))
        print(f"Extracted {len(old_spk_word_order)} ordered elements from old pkl.")
    else:
        print("Warning: layer 24 not found in old pkl.")
        old_spk_word_order = None
else:
    print("Warning: Old pkl not found, will fallback to alphabetical order.")
    old_spk_word_order = None

def get_all_layers(h5_path):
    layers = set()
    with h5py.File(h5_path, "r") as h5f:
        first_speaker = list(h5f.keys())[0]
        first_word = list(h5f[first_speaker].keys())[0]
        for layer_key in h5f[first_speaker][first_word].keys():
            layers.add(layer_key)
    return list(layers)

def process_single_layer(layer_name, input_h5):
    all_frames = []
    metadata = [] 
    
    with h5py.File(input_h5, "r") as h5f:
        if old_spk_word_order is not None:
            # First, process exactly according to the old ordering
            processed_pairs = set()
            for spk, word in old_spk_word_order:
                actual_word = word
                if spk == 'if6' and word == 'death' and 'gave' in h5f.get(spk, {}).keys() and 'death' not in h5f.get(spk, {}).keys():
                    actual_word = 'gave'
                elif spk == 'if6' and word == 'gave' and 'death' in h5f.get(spk, {}).keys() and 'gave' not in h5f.get(spk, {}).keys():
                    actual_word = 'death'
                    
                if spk in h5f and actual_word in h5f[spk] and layer_name in h5f[spk][actual_word]:
                    feat_matrix = h5f[spk][actual_word][layer_name][:]
                    T = feat_matrix.shape[0]
                    all_frames.append(feat_matrix)
                    metadata.extend([(spk, actual_word, t) for t in range(T)])
                    processed_pairs.add((spk, actual_word))
                    
            # In case the new h5 has MORE speakers/words than the old pkl, append them at the end
            for speaker_id in h5f.keys():
                for word_name in h5f[speaker_id].keys():
                    if (speaker_id, word_name) not in processed_pairs:
                        if layer_name in h5f[speaker_id][word_name]:
                            feat_matrix = h5f[speaker_id][word_name][layer_name][:]
                            T = feat_matrix.shape[0]
                            all_frames.append(feat_matrix)
                            metadata.extend([(speaker_id, word_name, t) for t in range(T)])
        else:
            # Fallback to alphabetical order
            for speaker_id in h5f.keys():
                for word_name in h5f[speaker_id].keys():
                    if layer_name in h5f[speaker_id][word_name]:
                        feat_matrix = h5f[speaker_id][word_name][layer_name][:]
                        T = feat_matrix.shape[0]
                        all_frames.append(feat_matrix)
                        metadata.extend([(speaker_id, word_name, t) for t in range(T)])
                        
    if len(all_frames) == 0:
        return layer_name, {}
        
    X_concat = np.vstack(all_frames)
    
    # Use exact TSNE parameters from old code
    tsne = TSNE(n_components=TSNE_DIM, random_state=42, init="pca", learning_rate="auto", n_jobs=1)
    X_tsne = tsne.fit_transform(X_concat)
    
    output_dict = {}
    current_idx = 0
    
    for i, frames in enumerate(all_frames):
        T = frames.shape[0]
        spk, word = metadata[current_idx][:2]
        
        if spk not in output_dict:
            output_dict[spk] = {}
            
        output_dict[spk][word] = X_tsne[current_idx:current_idx+T, :]
        current_idx += T
        
    return layer_name, output_dict

def run_tsne_on_features():
    for config in MODEL_CONFIGS:
        feat_h5 = config["feat_output"]
        tsne_h5 = config["tsne_output"].replace(".h5", "_random.h5")
        
        if not os.path.exists(feat_h5):
            print(f"File {feat_h5} not found. Skip.")
            continue
            
        print(f"\\nProcessing t-SNE for {feat_h5} -> {tsne_h5}")
        layers = get_all_layers(feat_h5)
        print(f"Found {len(layers)} layers: {layers}")
        
        start_time = time.time()
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        print(f"Running t-SNE in parallel on {num_cores} cores...")
        
        results = Parallel(n_jobs=num_cores)(
            delayed(process_single_layer)(layer, feat_h5) for layer in layers
        )
        
        print(f"Saving results to {tsne_h5}...")
        with h5py.File(tsne_h5, "w") as f_out:
            for layer_name, layer_dict in results:
                for spk, words in layer_dict.items():
                    for word, matrix in words.items():
                        group_path = f"{spk}/{word}/{layer_name}"
                        f_out.create_dataset(group_path, data=matrix, compression="gzip")
                        
        elapsed = time.time() - start_time
        print(f"Finished {tsne_h5} in {elapsed:.2f} seconds.")

# if __name__ == "__main__":
#     run_tsne_on_features()
'''
            cell['source'] = new_src.splitlines(True)

with open(out_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook successfully created!')
