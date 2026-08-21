import json
import os

path = r'c:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\glmm_prediction\test_nygaard_visualization.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The last cell should be the code cell, the second to last should be the markdown cell we added
if nb['cells'][-1]['cell_type'] == 'code' and 'Appendix: MFCC and STRF Extraction' in ''.join(nb['cells'][-2].get('source', [])):
    nb['cells'].pop()
    nb['cells'].pop()
    print('Removed previous injected cells.')

content = """# =========================================================
# 完整的 MFCC/STRF 基线特征提取及 GLMM 计算 (包含所有依赖)
# =========================================================
import os
import pandas as pd
import numpy as np
import h5py
import textgrid
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn.functional as F
import warnings
import traceback
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

# R setup
os.environ['R_HOME'] = 'C:\\\\Program Files\\\\R\\\\R-4.4.1'
os.environ["PATH"] += os.pathsep + r"C:\\Program Files\\R\\R-4.4.1\\bin\\x64"
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula, pandas2ri
import rpy2.robjects as ro
pandas2ri.activate()
warnings.filterwarnings('ignore')

# ----------------- Helper Functions -----------------
def get_pathset(paths):
    return [os.path.join(dir, each_file) for dir, mid, files in os.walk(paths) for each_file in files if each_file.endswith(".wav")]

def get_keywords_dict(df):
    keywords_dict={}
    for each_ in df.values:
        sentenceID=each_[df.columns.get_loc("SentenceID")]
        if sentenceID not in keywords_dict:
            keywords_dict[sentenceID]=[]
        keyword=each_[df.columns.get_loc("Keyword")]
        if keyword not in keywords_dict[sentenceID]:
            keywords_dict[sentenceID].append(keyword)
    return dict(sorted(keywords_dict.items()))

def get_keywords_list(df):
    out_dict={}
    for each_ in df.values:
        keyword_loc=df.columns.get_loc("Keyword")
        key_word = each_[keyword_loc]
        sentenceID = each_[df.columns.get_loc("SentenceID")]
        if sentenceID not in out_dict.keys():
            out_dict[sentenceID]=[]
        if key_word not in out_dict[sentenceID]:
            out_dict[sentenceID].append(key_word)
    return out_dict

def get_exposure_set(feature_dict, df_all, trainingTalkerID, sentenceID, key_word):
    set1_list=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]
    set2_list=[17,18,19,20,21,22,24,25,26,27,28,29,30,31,37,40]
    keywors_list = get_keywords_list(df_all)
    feats=[]
    for talker in trainingTalkerID:
        try:
            sentence_ind=(set1_list+set2_list).index(int(sentenceID[-3:])-1)
            key_word_ind=keywors_list[sentenceID].index(key_word)
            feats.append(feature_dict[talker][sentence_ind][key_word_ind])
        except:
            pass
    return feats

def get_test_feature(feature_dict, df_all, test_talker, sentenceID, key_word):
    set1_list=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]
    set2_list=[17,18,19,20,21,22,24,25,26,27,28,29,30,31,37,40]
    keywors_list = get_keywords_list(df_all)
    sentence_ind=(set1_list+set2_list).index(int(sentenceID[-3:])-1)
    key_word_ind=keywors_list[sentenceID].index(key_word)
    return feature_dict[test_talker][sentence_ind][key_word_ind]

def get_training_paths(TrainingTalkerID):
    TalkerID=[]
    for each_ID in TrainingTalkerID.split(", "):
        if each_ID[:3]=="CMN":
            TalkerID.append(f"ALL_{each_ID[-3:]}_M_CMN")
        else:
            TalkerID.append(f"ALL_{each_ID[-3:]}_M_ENG")
    return TalkerID

from numba import njit
@njit
def cosine_distance(vec1, vec2):
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for m in range(len(vec1)):
        dot += vec1[m] * vec2[m]
        norm1 += vec1[m]**2
        norm2 += vec2[m]**2
    if norm1 == 0 or norm2 == 0: return 1.0
    return 1.0 - (dot / ((norm1**0.5) * (norm2**0.5)))

@njit
def dtw_raw_distance(seq1, seq2, tau):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if tau == 2:
                cost = 0.0
                for k in range(len(seq1[i-1])):
                    cost += (seq1[i-1][k] - seq2[j-1][k])**2
                cost = cost**0.5
            else:
                cost = cosine_distance(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / ((n + m) / 2)

def standardization(sentences):
    all_frames = [frame for sent in sentences for seg in sent for frame in seg]
    reduced = np.asarray(all_frames, dtype=np.float32)
    mu = reduced.mean(axis=0, keepdims=True)
    sd = reduced.std(axis=0, keepdims=True) + 1e-8
    reduced = (reduced - mu) / sd
    count = 0
    filled = [[] for _ in range(len(sentences))]
    for i, sent in enumerate(sentences):
        new_segs = []
        for seg in sent:
            T = seg.shape[0]
            new_segs.append(reduced[count:count+T])
            count += T
        filled[i] = new_segs
    return filled

def create_set(audio_dir, df, reduced_data):
    import textgrid
    set1_list=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]
    set2_list=[17,18,19,20,21,22,24,25,26,27,28,29,30,31,37,40]
    
    keywords_dict=get_keywords_dict(df)
    keywords=[j for i in list(keywords_dict.values()) for j in i]
    audio_path=get_pathset(audio_dir)[::-1]
    out_dict={}
    word_features=[[] for i in range(len(keywords))]
    
    for __, each_path in enumerate(audio_path):
        current_talker=os.path.basename(each_path)[:13]
        if current_talker not in out_dict.keys():
            out_dict[current_talker]=[[] for i in range(32)]
        
        try:
            tg = textgrid.TextGrid.fromFile(each_path[:-3]+"TextGrid")
            tg_sentence = tg[0]
            for _,i in enumerate(tg[0]):
                if i.mark!="" and _ > 0: tg_sentence[_-1].maxTime=tg_sentence[_].minTime
            tg_sentence = [i for i in tg_sentence if i.mark!=""]
            tg_sentence=[tg_sentence[i] for i in set1_list+set2_list]
            tg_word = [i for i in tg[1] if i.mark!="" and i.mark!="sp"]
            
            count=0
            for _,each_sentence in enumerate(tg_sentence):
                sentence_total_length=each_sentence.maxTime-each_sentence.minTime
                for key_word in list(keywords_dict.values())[_]:
                    found=False
                    for each_word_tg in tg_word:
                        if each_word_tg.mark.lower()==key_word:
                            if each_word_tg.minTime >= each_sentence.minTime and each_word_tg.maxTime <= each_sentence.maxTime:
                                start=each_word_tg.minTime; end=each_word_tg.maxTime; found=True; break
                    
                    if found:
                        word_cut_start=start-each_sentence.minTime
                        word_cut_end=end-each_sentence.minTime
                        total_frames = reduced_data[_][__].shape[0]
                        word_start=round(total_frames * word_cut_start/sentence_total_length)
                        word_end=round(total_frames * word_cut_end/sentence_total_length)
                        features=reduced_data[_][__][word_start:word_end,:] 
                        word_features[count].append(features)
                        out_dict[current_talker][_].append(features)
                    count+=1
        except:
            continue
    return word_features, out_dict

def precompute_layer_distances(human_result_1a_split, audio_dir, reduced_data, tau, layer_name, func="mean"):
    word_features, feature_dict = create_set(audio_dir, human_result_1a_split, reduced_data)
    train_set_cache = {}
    test_word_cache = {}
    rows = []
    
    iterator = human_result_1a_split.to_dict('records')
    
    for row in iterator:
        kw = row["Keyword"]
        sent_id = row["SentenceID"]
        tr_ids = row["TrainingTalkerID"]
        test_file = os.path.basename(row["Filename"])[:13]
        trainingTalkerID = get_training_paths(tr_ids)
        
        cache_key = (tuple(sorted(trainingTalkerID)), sent_id, kw)
        if cache_key not in train_set_cache:
            train_feats = get_exposure_set(feature_dict, human_result_1a_split, trainingTalkerID, sent_id, kw)
            train_set_cache[cache_key] = train_feats
        else:
            train_feats = train_set_cache[cache_key]
            
        test_key = (test_file, sent_id, kw)
        if test_key not in test_word_cache:
            test_feat = get_test_feature(feature_dict, human_result_1a_split, test_file, sent_id, kw)
            test_word_cache[test_key] = test_feat
        else:
            test_feat = test_word_cache[test_key]

        dists = []
        for tr_feat in train_feats:
            if len(tr_feat) > 0 and len(test_feat) > 0:
                d = dtw_raw_distance(tr_feat, test_feat, tau)
                dists.append(d)
        
        if not dists: continue
            
        agg_dist = np.mean(dists)
        rows.append({
            'Keyword': kw, 'Condition2': row["Condition2"], 'TrainingTalkerID': tr_ids,
            'TestTalkerID': row["TestTalkerID"], 'SentenceID': sent_id, 'IsCorrect': row["IsCorrect"],
            'raw_distance': agg_dist, 'fold': row['fold']
        })
    return pd.DataFrame(rows)

def run_glmm_logic(k, train_df, test_df, purpose='optimize'):
    try:
        train_work = train_df.copy()
        if 'TrainingTalkerID1' not in train_work.columns:
            train_work['TrainingTalkerID1'] = train_work['TrainingTalkerID'].apply(lambda x: ",".join(sorted(x.split(", "))))
        train_work['similarity'] = np.exp(-train_work['raw_distance'] * k)

        train_agg = train_work.groupby(
            ['Keyword', 'Condition2', 'TrainingTalkerID1', 'TestTalkerID', 'SentenceID'], as_index=False
        ).agg(
            IsCorrect=('IsCorrect', 'mean'), similarity=('similarity', 'mean'),
            numCorrect=('IsCorrect', lambda x: (x == 1).sum()), numIncorrect=('IsCorrect', lambda x: (x == 0).sum())
        )

        train_mean = train_agg['similarity'].mean()
        train_sd   = train_agg['similarity'].std()
        if train_sd == 0: return 999.0 if purpose == 'optimize' else None

        ro.globalenv['r_train'] = pandas2ri.py2rpy(train_agg)
        ro.r('''
            library(lme4)
            r_train$SentenceID   <- factor(r_train$SentenceID)
            r_train$Keyword      <- factor(r_train$Keyword)
            r_train$TestTalkerID <- factor(r_train$TestTalkerID)
            model_train <- tryCatch({
                glmer(cbind(numCorrect, numIncorrect) ~ 1 + similarity + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
                      data=r_train, family=binomial(link="logit"), control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=10000)))
            }, error = function(e) { NULL })
            
            if (!is.null(model_train)) {
                z_train  <- summary(model_train)$coefficients[2, 3]
                ll_train <- as.numeric(logLik(model_train))
            } else {
                z_train <- -999
                ll_train <- -999
            }
        ''')
        z_train = ro.globalenv['z_train'][0]
        if purpose == 'optimize': return -z_train

        if purpose == 'evaluate' and test_df is not None:
            test_work = test_df.copy()
            if 'TrainingTalkerID1' not in test_work.columns:
                test_work['TrainingTalkerID1'] = test_work['TrainingTalkerID'].apply(lambda x: ",".join(sorted(x.split(", "))))
            test_work['similarity'] = np.exp(-test_work['raw_distance'] * k)

            test_agg = test_work.groupby(['Keyword', 'Condition2', 'TrainingTalkerID1', 'TestTalkerID', 'SentenceID'], as_index=False).agg(
                IsCorrect=('IsCorrect', 'mean'), similarity=('similarity', 'mean'),
                numCorrect=('IsCorrect', lambda x: (x == 1).sum()), numIncorrect=('IsCorrect', lambda x: (x == 0).sum())
            )
            ro.globalenv['r_test'] = pandas2ri.py2rpy(test_agg)
            ro.r('''
                r_test$SentenceID   <- factor(r_test$SentenceID)
                r_test$Keyword      <- factor(r_test$Keyword)
                r_test$TestTalkerID <- factor(r_test$TestTalkerID)
                model_test <- tryCatch({
                    glmer(cbind(numCorrect, numIncorrect) ~ 1 + similarity + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
                          data=r_test, family=binomial(link="logit"), control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=10000)))
                }, error = function(e) { NULL })
                
                if (!is.null(model_test)) {
                    ll_test <- as.numeric(logLik(model_test))
                    z_test  <- summary(model_test)$coefficients[2, 3]
                } else {
                    z_test <- -999
                    ll_test <- -999
                }
            ''')
            return {
                'z_train': z_train, 'z_test': ro.globalenv['z_test'][0],
                'poll_train': ro.globalenv['ll_train'][0] / (train_agg['numCorrect'].sum() + train_agg['numIncorrect'].sum()),
                'poll_test': ro.globalenv['ll_test'][0] / (test_agg['numCorrect'].sum() + test_agg['numIncorrect'].sum()),
            }
    except Exception:
        return 999.0 if purpose == 'optimize' else None
    return None

# ----------------- MFCC & STRF Extraction -----------------
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
            pass
            
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
            pass
            
    return layer_data

# ----------------- 修正后的 GLMM 拟合逻辑 -----------------
def process_memory_layer(layer_key, df_final, audio_dir, layer_data):
    try:
        std_data = standardization(layer_data)
        df_dist = precompute_layer_distances(df_final, audio_dir, std_data, 2, layer_key, func="mean")
        folds = sorted(df_dist['fold'].unique())
        
        diagnostic_results = []
        best_ks = []
        
        for f in folds:
            train_df = df_dist[df_dist['fold'] != f].copy()
            test_df  = df_dist[df_dist['fold'] == f].copy()
            res = minimize_scalar(lambda k: run_glmm_logic(k, train_df, None, 'optimize'), bounds=(0.001, 5.0), method='bounded')
            best_k = res.x
            best_ks.append(best_k)

        mean_k = np.mean(best_ks)
        corrected_results = []
        for f in folds:
            train_df = df_dist[df_dist['fold'] != f].copy()
            test_df  = df_dist[df_dist['fold'] == f].copy()
            metrics = run_glmm_logic(mean_k, train_df, test_df, 'evaluate')
            if metrics:
                corrected_results.append({
                    'layer': layer_key, 'fold': f, 'type': 'corrected', 'k': mean_k,
                    'z_train': metrics['z_train'], 'z_test':  metrics['z_test'],
                    'poll_train': metrics['poll_train'], 'poll_test':  metrics['poll_test']
                })
                
        return corrected_results
    except Exception as e:
        print(f"Error in {layer_key}: {traceback.format_exc()}")
        return None

# =========================================================
# 准备数据并运行
# =========================================================
audio_dir = r"../data/raw_data/xie21"
human_result_path = r"../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
if not os.path.exists(human_result_path):
    human_result_path = r"../data/raw_data/xie21/test.xlsx"

if os.path.exists(human_result_path):
    human_result = pd.read_excel(human_result_path)
    if "Experiment" in human_result.columns:
        human_result_1a = human_result[human_result["Experiment"]=="1a"].copy()
    else:
        human_result_1a = human_result.copy()
    
    if "TrainingTalkerID" in human_result_1a.columns:
        human_result_1a["TrainingTalkerID1"] = human_result_1a["TrainingTalkerID"].astype(str).apply(lambda x: ",".join(sorted(x.split(", "))) if pd.notna(x) else x)
        
    df = human_result_1a.copy()
    if 'WorkerID' in df.columns:
        participants = df[['WorkerID', 'TrainingTestSet', 'Condition2', 'TestTalkerID']].drop_duplicates().reset_index(drop=True)
        participants['combined_key'] = participants['TrainingTestSet'].astype(str) + "_" + participants['Condition2'].astype(str) + "_" + participants['TestTalkerID'].astype(str)
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        participants['fold'] = -1
        for fold_idx, (train_index, test_index) in enumerate(skf.split(participants, participants['combined_key'])):
            participants.loc[test_index, 'fold'] = fold_idx + 1
        df_final = df.merge(participants[['WorkerID', 'fold']], on='WorkerID', how='left')
    else:
        # Fallback if Nygaard dataset structure is slightly different
        df_final = df.copy()

    print("Data loaded. Starting Extraction...")
    
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

    valid_results = [r for sublist in baseline_results if sublist is not None for r in sublist]
    if valid_results:
        baseline_df = pd.DataFrame(valid_results)
        print(baseline_df)
        print("Baseline GLMM complete!")
else:
    print(f"Data file not found: {human_result_path}")
"""

new_cell_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['### Appendix: MFCC and STRF Extraction (Self-Contained)\\n', 'The following cell contains the fully self-contained extraction logic for MFCC and STRF features, including all necessary helper functions, CV fold logic, and data loading from `xie_glmm_cosine.ipynb`.\n', '> **Note:** Running this cell requires `torchaudio`, `rpy2`, `numba`, and will perform computationally heavy GLMM fitting using R.']
}
new_cell_code = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [line + '\n' for line in content.split('\n')]
}
nb['cells'].append(new_cell_md)
nb['cells'].append(new_cell_code)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Updated test_nygaard_visualization.ipynb with full dependencies')
