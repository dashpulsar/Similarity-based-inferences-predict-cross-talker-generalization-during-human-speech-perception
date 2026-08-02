#!/usr/bin/env python
# coding: utf-8

# # Variability Analysis for xie21 t-SNE Dataset (FT Model)
# 
# This notebook applies the corrected formulas for Generalized Variance and Hierarchical Aggregation.

# In[20]:


import pandas as pd
import numpy as np
import os
import h5py
import warnings
import traceback
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.1'
os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.4.1\bin\x64"
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
pandas2ri.activate()
warnings.filterwarnings('ignore')


# In[21]:


def load_single_layer_from_h5(h5_path, audio_dir, layer_key):
    def get_pathset(paths):
        return [os.path.join(dir, each_file) for dir, mid, files in os.walk(paths) for each_file in files if each_file.endswith(".wav")]
    audio_paths = get_pathset(audio_dir)[::-1]
    speakers = [os.path.basename(p).replace(".wav", "") for p in audio_paths]
    set1_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]
    set2_list = [17,18,19,20,21,22,24,25,26,27,28,29,30,31,37,40]
    combined_set = set1_list + set2_list
    
    layer_data = [ [] for _ in range(32) ]
    with h5py.File(h5_path, 'r') as h5f:
        root_keys = list(h5f.keys())
        is_layer_first = layer_key in root_keys
        
        for s_idx_mapped, s_idx_original in enumerate(combined_set):
            sent_key = f"sentence_{s_idx_original:02d}"
            for spk in speakers:
                if is_layer_first:
                    feat = h5f[layer_key][spk][sent_key][:]
                else:
                    feat = h5f[spk][sent_key][layer_key][:]
                layer_data[s_idx_mapped].append(feat)
    return layer_data, speakers

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


# In[22]:


def _generalized_variance(diff, tau):
    if diff.ndim == 1:
        return np.sum(np.abs(diff) ** tau)
    return np.sum(np.abs(diff) ** tau, axis=1)

def get_training_paths(TrainingTalkerID):
    TalkerID = []
    if not isinstance(TrainingTalkerID, str): return []
    for each_ID in TrainingTalkerID.split(", "):
        if each_ID[:3] == "CMN":
            TalkerID.append(f"ALL_{each_ID[-3:]}_M_CMN")
        else:
            TalkerID.append(f"ALL_{each_ID[-3:]}_M_ENG")
    return TalkerID

def find_talker_idx(talker, talker_index_map):
    for spk, idx in talker_index_map.items():
        if spk.startswith(talker):
            return idx
    return -1

def compute_variability_for_condition(training_talkers, sent_range, reduced_data, talker_index_map, tau, method):
    if method == 'WithinSentence':
        token_variances = []
        for each_talker in training_talkers:
            talker_idx = find_talker_idx(each_talker, talker_index_map)
            if talker_idx == -1: continue
            for sent_idx in sent_range:
                sent_frames = reduced_data[sent_idx][talker_idx]
                if sent_frames.shape[0] == 0: continue
                sent_mean = np.mean(sent_frames, axis=0)
                frame_vars = _generalized_variance(sent_frames - sent_mean, tau)
                token_variances.append(np.mean(frame_vars))
        if token_variances: return np.mean(token_variances)
        else: return np.nan

    elif method == 'BetweenSentence':
        type_means = []
        for sent_idx in sent_range:
            token_means = []
            for each_talker in training_talkers:
                talker_idx = find_talker_idx(each_talker, talker_index_map)
                if talker_idx == -1: continue
                sent_frames = reduced_data[sent_idx][talker_idx]
                if sent_frames.shape[0] > 0:
                    token_means.append(np.mean(sent_frames, axis=0))
            if token_means:
                type_means.append(np.mean(token_means, axis=0))
        if not type_means: return np.nan
        global_mean = np.mean(type_means, axis=0)
        type_vars = [_generalized_variance(tm - global_mean, tau) for tm in type_means]
        return np.mean(type_vars)

    elif method == 'Sentence_Order':
        token_variances = []
        for each_talker in training_talkers:
            talker_idx = find_talker_idx(each_talker, talker_index_map)
            if talker_idx == -1: continue
            for sent_idx in sent_range:
                sent_frames = reduced_data[sent_idx][talker_idx]
                if sent_frames.shape[0] < 2: continue
                diffs = sent_frames[1:] - sent_frames[:-1]
                frame_pair_vars = _generalized_variance(diffs, tau)
                token_variances.append(np.mean(frame_pair_vars))
        if token_variances: return np.mean(token_variances)
        else: return np.nan

def precompute_layer_variability(df, reduced_data, speakers, tau, method):
    talker_index_map = {spk: idx for idx, spk in enumerate(speakers)}
    unique_conditions = df[['TrainingTalkerID', 'TrainingTestSet']].drop_duplicates()
    var_map = {}
    for row in unique_conditions.itertuples(index=False):
        tr_id = row.TrainingTalkerID
        tr_set = str(row.TrainingTestSet)
        if pd.isna(tr_id): continue
        
        training_talkers = get_training_paths(tr_id)
        if tr_set.startswith("set1"):
            sent_range = range(16)
        else:
            sent_range = range(16, 32)
            
        var_val = compute_variability_for_condition(training_talkers, sent_range, reduced_data, talker_index_map, tau, method)
        var_map[(tr_id, tr_set)] = var_val
        
    rows = []
    for row in df.itertuples(index=False):
        var_val = var_map.get((getattr(row, 'TrainingTalkerID'), str(getattr(row, 'TrainingTestSet'))), np.nan)
        rows.append({
            'Keyword': getattr(row, 'Keyword'), 'Condition2': getattr(row, 'Condition2'),
            'TrainingTalkerID': getattr(row, 'TrainingTalkerID'), 'TestTalkerID': getattr(row, 'TestTalkerID'),
            'SentenceID': getattr(row, 'SentenceID'), 'IsCorrect': getattr(row, 'IsCorrect'),
            'variability': var_val, 'fold': getattr(row, 'fold')
        })
    return pd.DataFrame(rows)


# In[23]:


def run_glmm_variability_logic(tau, train_df, test_df, purpose='optimize'):
    try:
        train_work = train_df.copy()
        if 'TrainingTalkerID1' not in train_work.columns:
            train_work['TrainingTalkerID1'] = train_work['TrainingTalkerID'].apply(lambda x: ",".join(sorted(x.split(", "))))
        
        train_agg = train_work.groupby(
            ['Keyword', 'Condition2', 'TrainingTalkerID1', 'TestTalkerID', 'SentenceID'], as_index=False
        ).agg(
            IsCorrect=('IsCorrect', 'mean'), variability=('variability', 'mean'),
            numCorrect=('IsCorrect', lambda x: (x == 1).sum()), numIncorrect=('IsCorrect', lambda x: (x == 0).sum())
        )
        train_agg = train_agg.dropna(subset=['variability'])

        train_sd = train_agg['variability'].std()
        if train_sd == 0 or pd.isna(train_sd): return 999.0 if purpose == 'optimize' else None

        ro.globalenv['r_train'] = pandas2ri.py2rpy(train_agg)
        ro.r("""
            r_train$SentenceID   <- factor(r_train$SentenceID)
            r_train$Keyword      <- factor(r_train$Keyword)
            r_train$TestTalkerID <- factor(r_train$TestTalkerID)
            model_train <- glmer(
                cbind(numCorrect, numIncorrect) ~ 1 + variability + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
                data=r_train, family=binomial(link="logit"), control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=10000)))
            z_train  <- summary(model_train)$coefficients[2, 3]
            ll_train <- as.numeric(logLik(model_train))
        """)
        z_train = ro.globalenv['z_train'][0]
        if purpose == 'optimize': return -z_train

        if purpose == 'evaluate' and test_df is not None:
            test_work = test_df.copy()
            if 'TrainingTalkerID1' not in test_work.columns:
                test_work['TrainingTalkerID1'] = test_work['TrainingTalkerID'].apply(lambda x: ",".join(sorted(x.split(", "))))

            test_agg = test_work.groupby(['Keyword', 'Condition2', 'TrainingTalkerID1', 'TestTalkerID', 'SentenceID'], as_index=False).agg(
                IsCorrect=('IsCorrect', 'mean'), variability=('variability', 'mean'),
                numCorrect=('IsCorrect', lambda x: (x == 1).sum()), numIncorrect=('IsCorrect', lambda x: (x == 0).sum())
            )
            test_agg = test_agg.dropna(subset=['variability'])
            ro.globalenv['r_test'] = pandas2ri.py2rpy(test_agg)
            ro.r("""
                r_test$SentenceID   <- factor(r_test$SentenceID)
                r_test$Keyword      <- factor(r_test$Keyword)
                r_test$TestTalkerID <- factor(r_test$TestTalkerID)
                model_test <- glmer(
                    cbind(numCorrect, numIncorrect) ~ 1 + variability + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
                    data=r_test, family=binomial(link="logit"), control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=10000)))
                ll_test <- as.numeric(logLik(model_test))
                z_test  <- summary(model_test)$coefficients[2, 3]
            """)
            return {
                'z_train': z_train, 'z_test': ro.globalenv['z_test'][0],
                'poll_train': ro.globalenv['ll_train'][0] / (train_agg['numCorrect'].sum() + train_agg['numIncorrect'].sum()),
                'poll_test': ro.globalenv['ll_test'][0] / (test_agg['numCorrect'].sum() + test_agg['numIncorrect'].sum()),
            }
    except Exception:
        return 999.0 if purpose == 'optimize' else None
    return None

def process_single_layer_variability(layer_key, df_final, audio_dir, h5_path, method):
    try:
        layer_data, speakers = load_single_layer_from_h5(h5_path, audio_dir, layer_key)
        std_data = standardization(layer_data)
        folds = sorted(df_final['fold'].unique())
        diagnostic_results, best_taus = [], []

        for f in folds:
            train_idx = df_final['fold'] != f
            test_idx = df_final['fold'] == f
            
            def objective(tau_val):
                train_df = df_final[train_idx].copy()
                df_var = precompute_layer_variability(train_df, std_data, speakers, tau_val, method)
                return run_glmm_variability_logic(tau_val, df_var, None, 'optimize')
            
            res = minimize_scalar(objective, bounds=(0.5, 4.0), method='bounded')
            best_tau = res.x
            best_taus.append(best_tau)

            train_df = df_final[train_idx].copy()
            test_df  = df_final[test_idx].copy()
            train_var = precompute_layer_variability(train_df, std_data, speakers, best_tau, method)
            test_var  = precompute_layer_variability(test_df, std_data, speakers, best_tau, method)
            
            metrics = run_glmm_variability_logic(best_tau, train_var, test_var, 'evaluate')
            if metrics:
                diagnostic_results.append({
                    'layer': layer_key, 'fold': f, 'type': 'diagnostic', 'tau': best_tau,
                    'z_train': metrics['z_train'], 'z_test':  metrics['z_test'],
                    'poll_train': metrics['poll_train'], 'poll_test':  metrics['poll_test'],
                    'optimism': (metrics['poll_train'] - metrics['poll_test']) / abs(metrics['poll_train']),
                })

        mean_tau = np.mean(best_taus)
        corrected_results = []
        for f in folds:
            train_df = df_final[df_final['fold'] != f].copy()
            test_df  = df_final[df_final['fold'] == f].copy()
            train_var = precompute_layer_variability(train_df, std_data, speakers, mean_tau, method)
            test_var  = precompute_layer_variability(test_df, std_data, speakers, mean_tau, method)
            
            metrics = run_glmm_variability_logic(mean_tau, train_var, test_var, 'evaluate')
            if metrics:
                corrected_results.append({
                    'layer': layer_key, 'fold': f, 'type': 'corrected', 'tau': mean_tau,
                    'z_train': metrics['z_train'], 'z_test':  metrics['z_test'],
                    'poll_train': metrics['poll_train'], 'poll_test':  metrics['poll_test'],
                    'optimism': (metrics['poll_train'] - metrics['poll_test']) / abs(metrics['poll_train']),
                })

        results_df = pd.DataFrame(diagnostic_results + corrected_results)
        full_var_df = precompute_layer_variability(df_final, std_data, speakers, mean_tau, method)
        return results_df, layer_key, full_var_df
        
    except Exception as e:
        return f"EXCEPTION in {layer_key}: {e}\n{traceback.format_exc()}"

def run_variability_pipeline(layers_list, df_final, audio_dir, h5_path, method, n_jobs=-1):
    import time
    print(f"Starting {method} analysis for {len(layers_list)} layers...")
    start = time.time()
    
    tasks = [delayed(process_single_layer_variability)(k, df_final, audio_dir, h5_path, method) for k in layers_list]
    raw_results = Parallel(n_jobs=n_jobs, verbose=5)(tasks)

    valid_results, var_frames = [], {}
    for item in raw_results:
        if isinstance(item, str):
            print(item)
            continue
        if item is None: continue
        res_df, layer_key, var_df = item
        if res_df is not None and not res_df.empty: valid_results.append(res_df)
        if var_df is not None: 
            var_df = var_df[['Keyword', 'TrainingTalkerID', 'TestTalkerID', 'SentenceID', 'variability']].drop_duplicates().reset_index(drop=True)
            var_frames[layer_key] = var_df

    all_layers_results = pd.concat(valid_results, ignore_index=True) if valid_results else pd.DataFrame()
    df_all = df_final.copy()
    for layer_key, var_df in var_frames.items():
        df_all = df_all.merge(var_df.rename(columns={'variability': layer_key}), on=['Keyword', 'TrainingTalkerID', 'TestTalkerID', 'SentenceID'], how='left')

    print(f"Done in {(time.time() - start):.2f}s")
    return all_layers_results, df_all


# In[24]:


from sklearn.model_selection import StratifiedKFold

audio_dir = r"../data/raw_data/xie_liu_jaeger21/sound_stimuli"
human_result_path = r"../data/raw_data/xie_liu_jaeger21/X21-exposure-test-behavioral-data.xlsx"

# 1. 准备 Human Data
human_result = pd.read_excel(human_result_path)
human_result_1a = human_result[human_result["Experiment"]=="1a"].copy()
human_result_1a["TrainingTalkerID1"] = human_result_1a["TrainingTalkerID"].astype(str).apply(lambda x: ",".join(sorted(x.split(", "))) if pd.notna(x) else x)

df = human_result_1a.copy()
participants = df[['WorkerID', 'TrainingTestSet', 'Condition2', 'TestTalkerID']].drop_duplicates().reset_index(drop=True)
participants['combined_key'] = participants['TrainingTestSet'].astype(str) + "_" + participants['Condition2'].astype(str) + "_" + participants['TestTalkerID'].astype(str)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
participants['fold'] = -1
for fold_idx, (train_index, test_index) in enumerate(skf.split(participants, participants['combined_key'])):
    participants.loc[test_index, 'fold'] = fold_idx + 1
df_final = df.merge(participants[['WorkerID', 'fold']], on='WorkerID', how='left')

# 2. 自动检测并获取所有的 Layer 名称
h5_path = r"../preprocessing/xie21_tsne_3d_ft.h5"

with h5py.File(h5_path, 'r') as h5f:
    root_keys = list(h5f.keys())
    if 'cnn_6' in root_keys or 'tr_12' in root_keys:
        layers_list = root_keys
    else:
        first_spk = root_keys[0]
        first_sent = list(h5f[first_spk].keys())[0]
        layers_list = list(h5f[first_spk][first_sent].keys())

print(f"Found {len(layers_list)} layers to process: {layers_list}")

METHODS = ['BetweenSentence', 'WithinSentence', 'Sentence_Order']

for method in METHODS:
    print(f"\n=======================================================")
    print(f"Running GLMM Analysis for method: {method}")
    print(f"=======================================================")
    glmm_res, df_vals = run_variability_pipeline(layers_list, df_final, audio_dir, h5_path, method, n_jobs=-1)
    
    if not glmm_res.empty:
        glmm_res.to_csv(f"xie21_tsne_ft_variability_glmm_{method}.csv", index=False)
        df_vals.to_csv(f"xie21_tsne_ft_variability_values_{method}.csv", index=False)
        print(f"Results saved for {method}")


# In[26]:


glmm_res


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 假设你刚刚跑完了 WithinSentence 方法
method = "WithinSentence"
df_res = pd.read_csv(f"xie21_tsne_ft_variability_glmm_{method}.csv")

# 过滤出 corrected 的真实结果 (过滤掉 diagnostic 找参数的过程)
df_plot = df_res[df_res['type'] == 'corrected'].copy()

# 为了按顺序排列网络层，我们需要提取其中的数字
# layer 名称例如: 'cnn_2', 'cnn_3', ..., 'cnn_6' 或者 'tr_1', 'tr_12'
def extract_layer_num(layer_name):
    if 'cnn' in layer_name:
        return int(layer_name.split('_')[1])
    elif 'tr' in layer_name:
        return 6 + int(layer_name.split('_')[1]) # Transformer 层排在 CNN 后面
    return 0

df_plot['layer_order'] = df_plot['layer'].apply(extract_layer_num)
df_plot = df_plot.sort_values('layer_order')

plt.figure(figsize=(10, 6))
# 绘制折线图，x 轴是层名称，y 轴是测试集上的 Z-value
sns.lineplot(data=df_plot, x='layer', y='z_test', marker='o', markersize=8, linewidth=2, label='Z-value (Test)')

# 添加一条 y=0 的基准线
plt.axhline(0, color='red', linestyle='--', alpha=0.6)

plt.title(f"Predictive Power (Z-Value) Across Layers ({method})", fontsize=14)
plt.xlabel("Model Layers", fontsize=12)
plt.ylabel("Z-Value", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

