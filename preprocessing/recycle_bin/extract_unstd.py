import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

AUDIO_DIR = "../data/raw_data/alexander_nygaard19/sound_stimuli"
EXCEL_PATH = "../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
OUTPUT_H5 = "nygaard19_baseline_features_unstd.h5"

def get_strf_kernels():
    t = np.linspace(-0.2, 0.2, 41) 
    f = np.linspace(-1, 1, 21) 
    T_grid, F_grid = np.meshgrid(t, f)
    
    rates = [2, 4, 8, 16] 
    scales = [0.25, 0.5, 1.0] 
    
    kernels_real = []
    kernels_imag = []
    
    for r in rates:
        for s in scales:
            for d in [1, -1]: 
                env = np.exp(-0.5 * ((T_grid * r * 1.5)**2 + (F_grid * s * 1.5)**2))
                phase = 2 * np.pi * (r * T_grid + d * s * F_grid)
                c_real = env * np.cos(phase)
                c_imag = env * np.sin(phase)
                
                c_real -= c_real.mean()
                c_imag -= c_imag.mean()
                
                kernels_real.append(c_real)
                kernels_imag.append(c_imag)
                
    k_r = torch.tensor(np.array(kernels_real), dtype=torch.float32).unsqueeze(1)
    k_i = torch.tensor(np.array(kernels_imag), dtype=torch.float32).unsqueeze(1)
    return k_r, k_i

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

print("Loading behavioral data to filter required audio files...")
df = pd.read_excel(EXCEL_PATH)
used_filenames = df['FileName'].dropna().unique()
used_filenames_lower = set(f.lower() for f in used_filenames)
print(f"Found {len(used_filenames_lower)} unique audio files used in the experiment.")

audio_paths = []
for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.lower().endswith('.wav'):
            audio_paths.append(os.path.join(root, file))

print(f"Successfully located {len(audio_paths)} matching audio files in directory.")

# Pre-initialize transforms
mfcc_transform = T.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 23,
        "center": False
    }
)

mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80,
    center=False
)
k_r, k_i = get_strf_kernels()

# Temporary storage for features across all files
mfcc_all = []
strf_all = []
file_meta = []

for path in tqdm(audio_paths, desc="Extracting features"):
    import librosa
    wav, sr = librosa.load(path, sr=16000)
    wav = torch.tensor(wav).unsqueeze(0)
    sr = 16000
    if sr != 16000:
        
        
    # MFCC Extraction
    mfcc = mfcc_transform(wav)[0]
    delta1 = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta1)
    mfcc_feat = torch.cat([mfcc, delta1, delta2], dim=0).transpose(0, 1).numpy()
    
    # STRF Extraction
    mel = mel_transform(wav) 
    mel = torch.log(mel + 1e-6)
    mel_unsqueeze = mel.unsqueeze(0) 
    pad_h = k_r.shape[2] // 2
    pad_w = k_r.shape[3] // 2
    if mel_unsqueeze.shape[3] > pad_w and mel_unsqueeze.shape[2] > pad_h:
        mel_padded = F.pad(mel_unsqueeze, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    else:
        mel_padded = F.pad(mel_unsqueeze, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=float(mel_unsqueeze.min()))
    conv_r = F.conv2d(mel_padded, k_r) 
    conv_i = F.conv2d(mel_padded, k_i)
    strf_mag = torch.sqrt(conv_r**2 + conv_i**2)
    strf_feat = strf_mag.mean(dim=2).squeeze(0).transpose(0, 1).numpy()
    
    mfcc_all.append(mfcc_feat)
    strf_all.append(strf_feat)
    file_meta.append(os.path.basename(path))

print("Applying global standardization (Mean=0, Std=1)...")
mfcc_all_std = mfcc_all
strf_all_std = strf_all

print(f"Saving features to {OUTPUT_H5}...")
with h5py.File(OUTPUT_H5, "w") as h5f:
    grp_mfcc = h5f.create_group("MFCC")
    grp_strf = h5f.create_group("STRF")
    
    for i, meta in enumerate(file_meta):
        speaker_id = meta[:3].lower()
        word_name = meta.lower().split(" ")[-1][:-4]
        
        if speaker_id not in grp_mfcc:
            grp_mfcc.create_group(speaker_id)
        if speaker_id not in grp_strf:
            grp_strf.create_group(speaker_id)
            
        if word_name not in grp_mfcc[speaker_id]:
            grp_mfcc[speaker_id].create_dataset(word_name, data=mfcc_all_std[i], compression="gzip")
        if word_name not in grp_strf[speaker_id]:
            grp_strf[speaker_id].create_dataset(word_name, data=strf_all_std[i], compression="gzip")

print("Baseline feature extraction completed successfully!")

