import json
import traceback
path = r'c:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\glmm_prediction\xie_glmm_cosine.ipynb'
with open(path, 'r', encoding='utf-8') as f: nb = json.load(f)

# The new cell content
cell_content = '''# =========================================================
# 提取 MFCC 和 STRF 特征并在内存中处理
# =========================================================
import textgrid
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn.functional as F
import numpy as np
import os
import traceback
from joblib import Parallel, delayed

def extract_mfcc_features(audio_dir):
    set1_list=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]
    set2_list=[17,18,19,20,21,22,24,25,26,27,28,29,30,31,37,40]
    audio_path = get_pathset(audio_dir)[::-1]
    
    layer_data = [[] for _ in range(32)]
    
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
    
    print(f"Processing {len(audio_path)} audio files for MFCC extraction...")
    
    for __, each_path in enumerate(audio_path):
        try:
            tg = textgrid.TextGrid.fromFile(each_path[:-3] + "TextGrid")
            tg_sentence = tg[0]
            for _, i in enumerate(tg[0]):
                if i.mark != "" and _ > 0: 
                    tg_sentence[_-1].maxTime = tg_sentence[_].minTime
            tg_sentence = [i for i in tg_sentence if i.mark != ""]
            tg_sentence = [tg_sentence[i] for i in set1_list + set2_list]
            
            wav, sr = torchaudio.load(each_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                
            for _, each_sentence in enumerate(tg_sentence):
                start_time = each_sentence.minTime
                end_time = each_sentence.maxTime
                
                start_frame = int(start_time * 16000)
                end_frame = int(end_time * 16000)
                
                segment = wav[:, start_frame:end_frame]
                
                mfcc = mfcc_transform(segment)[0]
                delta1 = torchaudio.functional.compute_deltas(mfcc)
                delta2 = torchaudio.functional.compute_deltas(delta1)
                
                mfcc_features = torch.cat([mfcc, delta1, delta2], dim=0).transpose(0, 1).numpy()
                
                layer_data[_].append(mfcc_features)
        except Exception as e:
            print(f"Error processing {each_path}: {e}")
            
    return layer_data

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

def extract_strf_features(audio_dir):
    set1_list=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]
    set2_list=[17,18,19,20,21,22,24,25,26,27,28,29,30,31,37,40]
    audio_path = get_pathset(audio_dir)[::-1]
    
    layer_data = [[] for _ in range(32)]
    
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        center=False
    )
    
    k_r, k_i = get_strf_kernels()
    
    print(f"Processing {len(audio_path)} audio files for STRF extraction...")
    
    for __, each_path in enumerate(audio_path):
        try:
            tg = textgrid.TextGrid.fromFile(each_path[:-3] + "TextGrid")
            tg_sentence = tg[0]
            for _, i in enumerate(tg[0]):
                if i.mark != "" and _ > 0: 
                    tg_sentence[_-1].maxTime = tg_sentence[_].minTime
            tg_sentence = [i for i in tg_sentence if i.mark != ""]
            tg_sentence = [tg_sentence[i] for i in set1_list + set2_list]
            
            wav, sr = torchaudio.load(each_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                
            for _, each_sentence in enumerate(tg_sentence):
                start_time = each_sentence.minTime
                end_time = each_sentence.maxTime
                
                start_frame = int(start_time * 16000)
                end_frame = int(end_time * 16000)
                
                segment = wav[:, start_frame:end_frame]
                
                mel = mel_transform(segment) 
                mel = torch.log(mel + 1e-6)
                
                mel_unsqueeze = mel.unsqueeze(0) 
                
                pad_h = k_r.shape[2] // 2
                pad_w = k_r.shape[3] // 2
                mel_padded = F.pad(mel_unsqueeze, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
                
                conv_r = F.conv2d(mel_padded, k_r) 
                conv_i = F.conv2d(mel_padded, k_i)
                
                strf_mag = torch.sqrt(conv_r**2 + conv_i**2)
                
                segment_feats = strf_mag.mean(dim=2).squeeze(0).transpose(0, 1).numpy()
                layer_data[_].append(segment_feats)
        except Exception as e:
            print(f"Error processing {each_path}: {e}")
            
    return layer_data

def process_memory_layer(layer_key, df_final, audio_dir, layer_data):
    try:
        std_data = standardization(layer_data)
        # 强制将基线的 tau 设置为 2 (欧氏距离) 以符合原始论文逻辑，即使 notebook 设置了 cosine_distance
        df_dist = precompute_layer_distances(df_final, audio_dir, std_data, 2, layer_key, func="mean")
        train_df, test_df = prepare_data_for_glmm(df_dist)
        
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda k: run_glmm_logic(k, train_df, None, 'optimize'), bounds=(0.001, 5.0), method='bounded')
        best_k = res.x
        
        result = run_glmm_logic(best_k, train_df, test_df, 'evaluate')
        if result is None: return None
        result['layer'] = layer_key
        result['k'] = best_k
        result['type'] = 'corrected'
        return result
    except Exception as e:
        print(f"Error in {layer_key}: {traceback.format_exc()}")
        return None

# =========================================================
# 运行基线特征提取与 GLMM 分析
# =========================================================
mfcc_data = extract_mfcc_features(audio_dir)
print("MFCC extraction complete.")
strf_data = extract_strf_features(audio_dir)
print("STRF extraction complete.")

print("Running GLMM Analysis for MFCC and STRF...")
tasks = [
    delayed(process_memory_layer)('MFCC', df_final, audio_dir, mfcc_data),
    delayed(process_memory_layer)('STRF', df_final, audio_dir, strf_data)
]
baseline_results = Parallel(n_jobs=2, verbose=5)(tasks)

valid_results = [r for r in baseline_results if r is not None]
baseline_df = pd.DataFrame(valid_results)

all_layers_results_mfcc = baseline_df[baseline_df['layer'] == 'MFCC']
all_layers_results_strf = baseline_df[baseline_df['layer'] == 'STRF']

print("Baseline GLMM complete!")
'''

nb['cells'].insert(7, {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [line + '\n' for line in cell_content.split('\n')]
})

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Added baseline extraction cell!')
