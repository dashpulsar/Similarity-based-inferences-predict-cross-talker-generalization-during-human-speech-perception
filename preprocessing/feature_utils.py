import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

def standardize_features(feats_list):
    """
    Standardizes a list of (T, D) arrays across the entire corpus.
    Returns a list of standardized (T, D) arrays in the same order.
    """
    all_frames = np.vstack(feats_list)
    mu = all_frames.mean(axis=0, keepdims=True)
    sd = all_frames.std(axis=0, keepdims=True) + 1e-8
    
    standardized = []
    for f in feats_list:
        standardized.append((f - mu) / sd)
    return standardized


def get_all_layers(h5_path):
    layers = set()
    with h5py.File(h5_path, "r") as h5f:
        first_speaker = list(h5f.keys())[0]
        first_word = list(h5f[first_speaker].keys())[0]
        for layer_key in h5f[first_speaker][first_word].keys():
            layers.add(layer_key)
    return list(layers)


def process_single_layer(layer_name, input_h5):
    """Extract data for a single layer and perform t-SNE reduction (joblib worker function)"""
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


def _cleanup_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except:
            pass


def _extract_selected_layers(forward_module, input_values, layers_spec):
    cnn_handles, cnn_store = _register_cnn_hooks(forward_module)
    with torch.no_grad():
        out = forward_module(input_values, output_hidden_states=True, return_dict=True)
        tr_hidden = out.hidden_states
    _cleanup_hooks(cnn_handles)

    results = {}
    
    # Identify target temporal dimension based on the LAST CNN layer (index 6).
    # This precisely aligns all intermediate CNN layer dimensions to the dimension entering the Transformer.
    # Note: T_target matches tr_hidden[0].size(1) structurally.
    T_target = cnn_store[6].size(2)
    
    if "cnn" in layers_spec:
        for i in layers_spec["cnn"]:
            if i in cnn_store:
                feat = cnn_store[i]  # Original shape: (B, C, T_current)
                
                # Core Alignment Logic: Utilize 1D Adaptive Average Pooling to downsample T_current to T_target.
                if feat.size(2) != T_target:
                    feat = F.adaptive_avg_pool1d(feat, T_target)
                
                feat = feat.squeeze(0).permute(1, 0).contiguous() # Convert to (T_target, C)
                results[f"cnn_{i}"] = feat.cpu().numpy()
                
    if "tr" in layers_spec:
        for idx in layers_spec["tr"]:
            feat = tr_hidden[idx].squeeze(0) # (T_target, D)
            results[f"tr_{idx}"] = feat.cpu().numpy()
            
    return results


def _register_cnn_hooks(hubert_model):
    store = {}
    handles = []
    # Depending on the transformer version, conv_layers might be in different places
    try:
        conv_layers = hubert_model.feature_extractor.conv_layers
    except AttributeError:
        conv_layers = hubert_model.hubert.feature_extractor.conv_layers
        
    def make_hook(i):
        def hook(module, inputs, output):
            store[i] = output.detach()
        return hook
    for i, layer in enumerate(conv_layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    return handles, store


