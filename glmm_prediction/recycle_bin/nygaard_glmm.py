#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import os
import time
import warnings
import re
from joblib import Parallel, delayed
import h5py
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.optimize import minimize_scalar

# Set up R environment (adjust paths if necessary for your local setup)
os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.4.1'
os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.4.1\bin\x64"

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula, pandas2ri
pandas2ri.activate()
warnings.filterwarnings('ignore')


# In[21]:


@njit
def weighted_minkowski(vec1, vec2, tau, w=1):
    total = 0.0
    for m in range(len(vec1)):
        diff = w * abs(vec1[m] - vec2[m])
        total += (diff ** tau)
    return total**(1/tau)

@njit
def cosine_distance(vec1, vec2, tau=None):
    # tau is ignored for cosine, kept for compatibility
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
def dtw_raw_distance(seq1, seq2, tau=2.0, distance_type=0):
    # distance_type: 0 for minkowski, 1 for cosine
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if distance_type == 0:
                cost = weighted_minkowski(seq1[i-1], seq2[j-1], tau)
            else:
                cost = cosine_distance(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / ((n + m) / 2)


# In[22]:


def load_h5_data(h5_path):
    """
    Loads the precomputed features into memory to avoid concurrent HDF5 read issues.
    Format returned: dict[layer_name][speaker_id][word_name] = ndarray
    """
    data_dict = {}
    with h5py.File(h5_path, 'r') as h5f:
        root_keys = list(h5f.keys())
        if len(root_keys) > 0 and (root_keys[0].startswith('cnn_') or root_keys[0].startswith('tr_') or root_keys[0] in ['MFCC', 'STRF']):
            for layer in h5f.keys():
                data_dict[layer] = {}
                for spk in h5f[layer].keys():
                    data_dict[layer][spk] = {}
                    for word in h5f[layer][spk].keys():
                        data_dict[layer][spk][word] = h5f[layer][spk][word][:]
        else:
            for spk in h5f.keys():
                for word in h5f[spk].keys():
                    for layer in h5f[spk][word].keys():
                        if layer not in data_dict:
                            data_dict[layer] = {}
                        if spk not in data_dict[layer]:
                            data_dict[layer][spk] = {}
                        data_dict[layer][spk][word] = h5f[spk][word][layer][:]
    return data_dict

def get_training_talkers_map(df):
    """Subject -> [Training Speakers] map for Nygaard."""
    subject_map = {}
    default_control_group = ['ef1','ef2','ef3','em1','em2','em3']
    
    for sub in df['Subject'].unique():
        learning_phase = df[(df['Subject'] == sub) & (df['Phase'] == 'Learning')]
        if not learning_phase.empty:
            # Extract speaker IDs from FileName (first 3 chars)
            spks = learning_phase['FileName'].dropna().apply(lambda x: str(x)[:3].lower()).unique()
            subject_map[sub] = list(spks)
        else:
            subject_map[sub] = default_control_group
    return subject_map

def create_nygaard_dataset(df_full):
    """
    Creates the stratified 3-fold split by grouping subjects with identical training/test files.
    """
    df_test = df_full[df_full["Phase"]=="Test"].copy().reset_index(drop=True)
    if 'TrainingAccent' not in df_test.columns and 'TrainingAccent' in df_full.columns:
        # Inherit from full if applicable
        df_test['TrainingAccent'] = df_full.groupby('Subject')['TrainingAccent'].transform('first')
    elif 'TrainingAccent' not in df_full.columns:
        df_test['TrainingAccent'] = "English"

    df_test['TrainingAccent'] = df_test['TrainingAccent'].fillna("English")
    
    df_test_TrainingFile = []
    df_test_TestFile = []
    for index, row in df_test.iterrows():
        current_participant = row["Subject"]
        df_test_TrainingFile.append("_".join(sorted(list(df_full[(df_full["Subject"]==current_participant) & (df_full["Phase"]=="Learning")]["FileName"].dropna().astype(str)))))
        df_test_TestFile.append("_".join(sorted(list(df_full[(df_full["Subject"]==current_participant) & (df_full["Phase"]=="Test")]["FileName"].dropna().astype(str)))))
        
    df_test["TrainingFile"] = df_test_TrainingFile
    df_test["TestFile"] = df_test_TestFile
    df_test["TrainingFile"] = df_test["TrainingFile"].apply(lambda x: "English" if x=="" else x)
    df_test["TrainingTestFile"] = df_test["TrainingFile"] + df_test["TestFile"]
    
    participants_df = df_test[['Subject', 'TrainingTestFile']].drop_duplicates().reset_index(drop=True)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    participants_df['fold'] = -1
    try:
        split_generator = skf.split(participants_df, participants_df['TrainingTestFile'])
        for fold_idx, (train_index, test_index) in enumerate(split_generator):
            participants_df.loc[test_index, 'fold'] = fold_idx + 1
    except ValueError as e:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for fold_idx, (train_index, test_index) in enumerate(kf.split(participants_df)):
            participants_df.loc[test_index, 'fold'] = fold_idx + 1
            
    df_test = df_test.merge(participants_df[['Subject', 'fold']], on='Subject', how='left')
    # Standardize column names for GLMM
    if 'Speaker_full' not in df_test.columns:
        if 'Accent' in df_test.columns and 'Speaker' in df_test.columns:
            df_test["Speaker_full"] = df_test["Accent"].astype(str) + df_test["Speaker"].astype(str)
        else:
            df_test["Speaker_full"] = df_test["FileName"].apply(lambda x: str(x)[:3].lower() if pd.notna(x) else "unknown")
    return df_test

def calculate_nygaard_distances(layer_name, layer_data, df_test, subject_map, distance_type="minkowski", tau=2.0):
    """Calculates distances for a single layer for all test trials."""
    results = []
    dtype_int = 0 if distance_type == "minkowski" else 1
    
    for row in df_test.itertuples():
        idx = row.Index
        fname = str(row.FileName)
        if pd.isna(row.FileName):
            continue
            
        test_speaker = fname[:3].lower()
        word_name = fname.lower().split(" ")[-1][:-4]
        
        if test_speaker not in layer_data or word_name not in layer_data[test_speaker]:
            continue
            
        test_feat = layer_data[test_speaker][word_name]
        train_talkers = subject_map.get(row.Subject, [])
        dists = []
        for t_spk in train_talkers:
            if t_spk in layer_data and word_name in layer_data[t_spk]:
                train_feat = layer_data[t_spk][word_name]
                d = dtw_raw_distance(test_feat, train_feat, tau=tau, distance_type=dtype_int)
                dists.append(d)
                
        if dists:
            row_dict = row._asdict()
            row_dict['layer'] = layer_name
            row_dict['raw_distance'] = np.mean(dists)
            results.append(row_dict)
            
    return pd.DataFrame(results)


# In[23]:


def fit_and_evaluate_split_nygaard(k, train_df, target_df, target_type='val'):
    """
    Fits GLMER on Train data with similarity_scaled = exp(-raw_distance * k).
    Validates on Target data using Training scaling params.
    """
    base = importr('base')
    stats = importr('stats')
    lme4 = importr('lme4')
    try:
        train_work = train_df.copy()
        train_work['similarity'] = np.exp(-train_work['raw_distance'] * k)
        
        groups = ['Keyword', 'TestTalker', 'SubjectID']
        if 'TrainingAccent' in train_work.columns: groups.append('TrainingAccent')
        
        train_agg = train_work.groupby(groups, as_index=False).agg(
            similarity=('similarity', 'mean'),
            numCorrect=('NumCorrect', 'sum'),
            numWord=('NumWord', 'sum')
        )
        train_agg['numIncorrect'] = (train_agg['numWord'] - train_agg['numCorrect']).clip(lower=0)
        
        sim_std = train_agg['similarity'].std()
        sim_mean = train_agg['similarity'].mean()
        
        # Check Variance (Optional filters can be placed here)
        if sim_std == 0: return -999
        
        train_agg['similarity_scaled'] = (train_agg['similarity'] - sim_mean) / (2 * sim_std)
        
        target_work = target_df.copy()
        target_work['similarity'] = np.exp(-target_work['raw_distance'] * k)
        
        target_agg = target_work.groupby(groups, as_index=False).agg(
            similarity=('similarity', 'mean'),
            numCorrect=('NumCorrect', 'sum'),
            numWord=('NumWord', 'sum')
        )
        target_agg['numIncorrect'] = (target_agg['numWord'] - target_agg['numCorrect']).clip(lower=0)
        target_agg['similarity_scaled'] = (target_agg['similarity'] - sim_mean) / (2 * sim_std)
        
        ro.globalenv['r_train'] = pandas2ri.py2rpy(train_agg)
        ro.globalenv['r_target'] = pandas2ri.py2rpy(target_agg)
        
        ro.r('''
            library(lme4)
            r_train$Keyword <- factor(r_train$Keyword)
            r_train$TestTalker <- factor(r_train$TestTalker)
            r_train$SubjectID <- factor(r_train$SubjectID)
            
            r_target$Keyword <- factor(r_target$Keyword)
            r_target$TestTalker <- factor(r_target$TestTalker)
            r_target$SubjectID <- factor(r_target$SubjectID)
            
            model_train <- tryCatch({
                glmer(cbind(numCorrect, numIncorrect) ~ similarity_scaled + (1 + similarity_scaled | SubjectID), 
                      data=r_train, family=binomial(link="logit"), 
                      control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
            }, error=function(e){ NULL })
            
            model_target <- tryCatch({
                glmer(cbind(numCorrect, numIncorrect) ~ similarity_scaled + (1 + similarity_scaled | SubjectID), 
                      data=r_target, family=binomial(link="logit"), 
                      control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
            }, error=function(e){ NULL })
            
            res_list <- list()
            if (!is.null(model_train) && !is.null(model_target)) {
                res_list$z_train <- summary(model_train)$coefficients[2,3]
                res_list$loglik_train <- as.numeric(logLik(model_train))
                
                res_list$z_target <- summary(model_target)$coefficients[2,3]
                res_list$loglik_target <- as.numeric(logLik(model_target))
                res_list$success <- TRUE
            } else {
                res_list$success <- FALSE
            }
            res_list
        ''')
        
        res_r = ro.globalenv['res_list']
        if not res_r.rx2('success')[0]: return None
            
        z_target = res_r.rx2('z_target')[0]
        
        if False:
            return None
            
        return {
            'z_train': res_r.rx2('z_train')[0],
            'loglik_train': res_r.rx2('loglik_train')[0],
            'n_train': train_agg['numWord'].sum(),
            'z_target': z_target,
            'loglik_target': res_r.rx2('loglik_target')[0],
            'n_target': target_agg['numWord'].sum()
        }
    except Exception as e:
        return -999

def objective_on_validation_nygaard(k, train_df, val_df, alpha):
    metrics = fit_and_evaluate_split_nygaard(k, train_df, val_df, target_type='val')
    if metrics == -999:
        return 9999.0 
    loss = -metrics['z_target'] + alpha * (k**2)
    return loss

def process_layer_nygaard_l2(layer_key, layer_df, alpha=0.1):
    try:
        work_df = layer_df.copy()
        rename_map = {
            'correct': 'Keyword',           
            'Speaker_full': 'TestTalker',   
            'accuracy': 'NumCorrect',       
            'Subject': 'SubjectID'          
        }
        work_df.rename(columns={k:v for k,v in rename_map.items() if k in work_df.columns}, inplace=True)
        if 'NumCorrect' not in work_df.columns: return None
        if 'NumWord' not in work_df.columns: work_df['NumWord'] = 1 
        work_df = work_df.dropna(subset=['raw_distance'])
        if len(work_df) == 0: return None
        
        if 'fold' not in work_df.columns:
             return None
             
        work_df['fold'] = work_df['fold'].astype(int)
        folds = sorted(work_df['fold'].unique())
        num_folds = len(folds)
        if num_folds < 3: return None
        
        results = []
        for i, test_fold in enumerate(folds):
            val_fold = folds[(i + 1) % num_folds]
            train_fold = folds[(i + 2) % num_folds]
            
            test_df = work_df[work_df['fold'] == test_fold].copy()
            val_df = work_df[work_df['fold'] == val_fold].copy()
            train_df = work_df[work_df['fold'] == train_fold].copy()
            
            res = minimize_scalar(
                lambda k: objective_on_validation_nygaard(k, train_df, val_df, alpha), 
                bounds=(0.001, 5.0), 
                method='bounded'
            )
            best_k = res.x
            
            final_metrics = fit_and_evaluate_split_nygaard(best_k, train_df, test_df, target_type='test')
            if final_metrics:
                poll_train = final_metrics['loglik_train'] / final_metrics['n_train']
                poll_test = final_metrics['loglik_target'] / final_metrics['n_target']
                opt = (poll_train - poll_test) / abs(poll_train)
                
                results.append({
                    'layer': layer_key, 
                    'fold': test_fold,
                    'type': 'corrected',
                    'k': best_k, 
                    'z_train': final_metrics['z_train'], 
                    'z_test': final_metrics['z_target'], 
                    'poll_train': poll_train, 
                    'poll_test': poll_test, 
                    'optimism': opt
                })
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error in Layer {layer_key}: {e}")
        return pd.DataFrame()


# In[24]:


def run_full_nygaard_analysis(df_behavioral, h5_path, distance_type="minkowski", tau=2.0, alpha=0.1, n_jobs=-1):
    df_test = create_nygaard_dataset(df_behavioral)
    subject_map = get_training_talkers_map(df_behavioral)
    
    if not os.path.exists(h5_path):
        print(f"File {h5_path} does not exist. Skipping...")
        return pd.DataFrame()
        
    print(f"Loading features from {h5_path}...")
    data_dict = load_h5_data(h5_path)
    
    # Restrict to specified subset of layers (CNNs: 2-6, Transformers: evens 0-24, plus baselines)
    valid_layers_set = {f"cnn_{i}" for i in range(2, 7)}.union({f"tr_{i}" for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]}).union({'MFCC', 'STRF'})
    available_layers = [l for l in list(data_dict.keys()) if l in valid_layers_set]
    
    print(f"Calculating {distance_type} distances for {len(available_layers)} layers...")
    dist_dfs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(calculate_nygaard_distances)(
            layer, data_dict[layer], df_test, subject_map, distance_type, tau
        ) for layer in available_layers
    )
    
    valid_dists = [d for d in dist_dfs if not d.empty]
    if not valid_dists:
        return pd.DataFrame()
    
    df_raw_dist_global = pd.concat(valid_dists, ignore_index=True)
    
    print(f"Running 3-fold GLMM analysis (alpha={alpha})...")
    tasks = []
    for layer in available_layers:
        layer_data = df_raw_dist_global[df_raw_dist_global['layer'] == layer].copy()
        if not layer_data.empty:
            tasks.append(delayed(process_layer_nygaard_l2)(layer, layer_data, alpha=alpha))
            
    results = Parallel(n_jobs=n_jobs, verbose=5)(tasks)
    
    valid_res = [r for r in results if r is not None and not r.empty]
    if not valid_res:
        return pd.DataFrame()
        
    df_final = pd.concat(valid_res, ignore_index=True)
    return df_final


# In[25]:


EXCEL_PATH = "../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
