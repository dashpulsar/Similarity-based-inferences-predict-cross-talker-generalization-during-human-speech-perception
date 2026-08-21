import json

with open('../preprocessing/feature_extraction_nygaard19.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

original_cell_content = """import h5py
import numpy as np
from joblib import Parallel, delayed
from sklearn.manifold import TSNE
import multiprocessing
import time

TSNE_DIM = 3

# Reuse MODEL_CONFIGS from previous cell
# It is assumed that MODEL_CONFIGS is defined in the workspace

def get_all_layers(h5_path):
    layers = set()
    with h5py.File(h5_path, "r") as h5f:
        first_speaker = list(h5f.keys())[0]
        first_word = list(h5f[first_speaker].keys())[0]
        for layer_key in h5f[first_speaker][first_word].keys():
            layers.add(layer_key)
    return list(layers)

def process_single_layer(layer_name, input_h5):
    \"\"\"Extract data for a single layer and perform t-SNE reduction (joblib worker function)\"\"\"
    all_frames = []
    metadata = [] 
    
    with h5py.File(input_h5, "r") as h5f:
        for speaker_id in h5f.keys():
            for word_name in h5f[speaker_id].keys():
                if layer_name in h5f[speaker_id][word_name]:
                    feat_matrix = h5f[speaker_id][word_name][layer_name][:]
                    T = feat_matrix.shape[0]
                    all_frames.append(feat_matrix)
                    metadata.append({"speaker": speaker_id, "word": word_name, "length": T})
                    
    if not all_frames:
        return layer_name, None, metadata

    flatten = np.vstack(all_frames)
    print(f"[{layer_name}] Initiating t-SNE computation... Total frames: {flatten.shape[0]}, Feature dimension: {flatten.shape[1]}")
    start_time = time.time()
    
    tsne = TSNE(n_components=TSNE_DIM, random_state=42, n_jobs=1)
    reduced_features = tsne.fit_transform(flatten)
    
    cost_time = time.time() - start_time
    print(f"[{layer_name}] t-SNE computation finalized. Elapsed time: {cost_time:.2f} seconds")
    
    # Return dimension-reduced results to the main process
    return layer_name, reduced_features, metadata

# ========= Initialize Parallel t-SNE Computation =========
max_workers = min(multiprocessing.cpu_count(), 30)
print(f"Initializing parallel processing framework via joblib. Allocated CPU cores: {max_workers}")

for config in MODEL_CONFIGS:
    input_h5 = config["feat_output"]
    output_h5 = config["tsne_output"]
    
    print(f"\\n--- Initiating Dimensionality Reduction for Configuration: {config['model_id']} ---")
    layers = get_all_layers(input_h5)
    print(f"Target layers for processing: {layers}")

    # Core execution: utilizing robust joblib concurrency
    # n_jobs specifies the degree of parallelism
    results = Parallel(n_jobs=max_workers)(
        delayed(process_single_layer)(layer, input_h5) for layer in layers
    )

    # Aggregate and write results to the output HDF5 structure
    print(f"\\nAggregating results and writing to {output_h5}...")
    with h5py.File(output_h5, "w") as out_h5:
        for layer_name, reduced_features, metadata in results:
            if reduced_features is None: 
                continue
            
            layer_group = out_h5.create_group(layer_name)
            current_idx = 0
            for meta in metadata:
                spk, word, length = meta["speaker"], meta["word"], meta["length"]
                word_3d = reduced_features[current_idx : current_idx + length]
                current_idx += length
                
                if spk not in layer_group:
                    layer_group.create_group(spk)
                layer_group[spk].create_dataset(word, data=word_3d)

    print(f"Dimensionality reduction completed successfully. Results stored in {output_h5}")
print("\\nPipeline execution fully completed.")
"""

new_cell_content = """import h5py
import numpy as np
from joblib import Parallel, delayed
from openTSNE import TSNE
from sklearn.decomposition import PCA
import multiprocessing
import time

TSNE_DIM = 3

# Reuse MODEL_CONFIGS from previous cell
# It is assumed that MODEL_CONFIGS is defined in the workspace

def get_all_layers(h5_path):
    layers = set()
    with h5py.File(h5_path, "r") as h5f:
        first_speaker = list(h5f.keys())[0]
        first_word = list(h5f[first_speaker].keys())[0]
        for layer_key in h5f[first_speaker][first_word].keys():
            layers.add(layer_key)
    return list(layers)

def process_single_layer(layer_name, input_h5):
    \"\"\"Extract data for a single layer and perform t-SNE reduction (joblib worker function)\"\"\"
    all_frames = []
    metadata = [] 
    
    with h5py.File(input_h5, "r") as h5f:
        for speaker_id in h5f.keys():
            for word_name in h5f[speaker_id].keys():
                if layer_name in h5f[speaker_id][word_name]:
                    feat_matrix = h5f[speaker_id][word_name][layer_name][:]
                    T = feat_matrix.shape[0]
                    all_frames.append(feat_matrix)
                    metadata.append({"speaker": speaker_id, "word": word_name, "length": T})
                    
    if not all_frames:
        return layer_name, None, metadata

    flatten = np.vstack(all_frames)
    print(f"[{layer_name}] Initiating t-SNE computation... Total frames: {flatten.shape[0]}, Feature dimension: {flatten.shape[1]}")
    start_time = time.time()
    
    # PCA down to 50 dimensions to vastly speed up t-SNE KNN computation
    if flatten.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        flatten_pca = pca.fit_transform(flatten)
    else:
        flatten_pca = flatten

    # Use openTSNE for extreme speedup. n_jobs=1 because Joblib already runs layers in parallel.
    tsne = TSNE(n_components=TSNE_DIM, random_state=42, n_jobs=1)
    reduced_features = tsne.fit(flatten_pca)
    
    cost_time = time.time() - start_time
    print(f"[{layer_name}] t-SNE computation finalized. Elapsed time: {cost_time:.2f} seconds")
    
    # Return dimension-reduced results to the main process
    return layer_name, reduced_features, metadata

# ========= Initialize Parallel t-SNE Computation =========
max_workers = min(multiprocessing.cpu_count(), 30)
print(f"Initializing parallel processing framework via joblib. Allocated CPU cores: {max_workers}")

for config in MODEL_CONFIGS:
    if "ft" not in config["model_id"]:
        print(f"\\n--- Skipping Dimensionality Reduction for Configuration: {config['model_id']} (Already computed) ---")
        continue

    input_h5 = config["feat_output"]
    output_h5 = config["tsne_output"]
    
    print(f"\\n--- Initiating Dimensionality Reduction for Configuration: {config['model_id']} ---")
    layers = get_all_layers(input_h5)
    print(f"Target layers for processing: {layers}")

    # Core execution: utilizing robust joblib concurrency
    # n_jobs specifies the degree of parallelism
    results = Parallel(n_jobs=max_workers)(
        delayed(process_single_layer)(layer, input_h5) for layer in layers
    )

    # Aggregate and write results to the output HDF5 structure
    print(f"\\nAggregating results and writing to {output_h5}...")
    with h5py.File(output_h5, "w") as out_h5:
        for layer_name, reduced_features, metadata in results:
            if reduced_features is None: 
                continue
            
            layer_group = out_h5.create_group(layer_name)
            current_idx = 0
            for meta in metadata:
                spk, word, length = meta["speaker"], meta["word"], meta["length"]
                word_3d = reduced_features[current_idx : current_idx + length]
                current_idx += length
                
                if spk not in layer_group:
                    layer_group.create_group(spk)
                layer_group[spk].create_dataset(word, data=word_3d)

    print(f"Dimensionality reduction completed successfully. Results stored in {output_h5}")
print("\\nPipeline execution fully completed.")
"""

# 1. Restore the last cell
nb['cells'][-1]['source'] = [line + '\n' for line in original_cell_content.split('\n')]
nb['cells'][-1]['outputs'] = [] # clear outputs to be safe
nb['cells'][-1]['execution_count'] = None

# 2. Append the new cell
new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "open_tsne_fast_cell",
   "metadata": {},
   "outputs": [],
   "source": [line + '\n' for line in new_cell_content.split('\n')]
}
nb['cells'].append(new_cell)

with open('../preprocessing/feature_extraction_nygaard19.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Restored old cell and appended new cell successfully!')
