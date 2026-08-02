import pandas as pd
import numpy as np
import h5py
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold
from numba import njit

import os
os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.4.1'
os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.4.1\bin\x64"

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula, pandas2ri
pandas2ri.activate()

@njit
def weighted_minkowski(vec1, vec2, tau, w=1):
    total = 0.0
    for m in range(len(vec1)):
        diff = w * abs(vec1[m] - vec2[m])
        total += (diff ** tau)
    return total**(1/tau)

@njit
def cosine_distance(vec1, vec2, tau=None):
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


def load_h5_data(h5_path):
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
    subject_map = {}
    default_control_group = ['ef1','ef2','ef3','em1','em2','em3']
    for sub in df['Subject'].unique():
        learning_phase = df[(df['Subject'] == sub) & (df['Phase'] == 'Learning')]
        if not learning_phase.empty:
            spks = learning_phase['FileName'].dropna().apply(lambda x: str(x)[:3].lower()).unique()
            subject_map[sub] = list(spks)
        else:
            subject_map[sub] = default_control_group
    return subject_map

def create_nygaard_dataset(df_full):
    df_test = df_full[df_full["Phase"]=="Test"].copy().reset_index(drop=True)
    if 'TrainingAccent' not in df_test.columns and 'TrainingAccent' in df_full.columns:
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
    if 'Speaker_full' not in df_test.columns:
        if 'Accent' in df_test.columns and 'Speaker' in df_test.columns:
            df_test["Speaker_full"] = df_test["Accent"].astype(str) + df_test["Speaker"].astype(str)
        else:
            df_test["Speaker_full"] = df_test["FileName"].apply(lambda x: str(x)[:3].lower() if pd.notna(x) else "unknown")
    return df_test

def calculate_nygaard_distances(layer_name, layer_data, df_test, subject_map, distance_type="minkowski", tau=2.0):
    results = []
    dtype_int = 0 if distance_type == "minkowski" else 1
    for row in df_test.itertuples():
        fname = str(row.FileName)
        if pd.isna(row.FileName): continue
        test_speaker = fname[:3].lower()
        word_name = fname.lower().split(" ")[-1][:-4]
        if test_speaker not in layer_data or word_name not in layer_data[test_speaker]: continue
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

def debug_fit_and_evaluate(k, train_df, target_df):
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
        
        if sim_std == 0: 
            print("SIM_STD is 0")
            return -999
        
        train_agg['similarity_scaled'] = (train_agg['similarity'] - sim_mean) / (2 * sim_std)
        
        ro.globalenv['r_train'] = pandas2ri.py2rpy(train_agg)
        
        ro.r('''
            library(lme4)
            r_train$Keyword <- factor(r_train$Keyword)
            r_train$TestTalker <- factor(r_train$TestTalker)
            r_train$SubjectID <- factor(r_train$SubjectID)
            
            tryCatch({
                glmer(cbind(numCorrect, numIncorrect) ~ 1 + similarity_scaled + (1|TestTalker) + (1|Keyword), 
                      data=r_train, family=binomial(link="logit"), 
                      control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
            }, error=function(e){ print(e); NULL })
        ''')
        print("GLMER executed!")
    except Exception as e:
        print("Python Exception:", e)

if __name__ == "__main__":
    EXCEL_PATH = "../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
    df_behavioral = pd.read_excel(EXCEL_PATH)
    df_test = create_nygaard_dataset(df_behavioral)
    subject_map = get_training_talkers_map(df_behavioral)
    
    h5_path = "../preprocessing/nygaard19_baseline_features.h5"
    data_dict = load_h5_data(h5_path)
    layer_name = 'MFCC'
    layer_data = data_dict[layer_name]
    
    dist_df = calculate_nygaard_distances(layer_name, layer_data, df_test, subject_map, distance_type="minkowski", tau=2.0)
    print("Dist DF shape:", dist_df.shape)
    
    rename_map = {
        'Word': 'Keyword',           
        'Speaker_full': 'TestTalker',   
        'accuracy': 'NumCorrect',       
        'Subject': 'SubjectID'          
    }
    work_df = dist_df.rename(columns=rename_map)
    work_df['NumWord'] = 1
    work_df = work_df.dropna(subset=['raw_distance'])
    work_df['fold'] = work_df['fold'].astype(int)
    
    folds = sorted(work_df['fold'].unique())
    print("Folds:", folds)
    if len(folds) >= 3:
        test_fold = folds[0]
        val_fold = folds[1]
        train_fold = folds[2]
        
        train_df = work_df[work_df['fold'] == train_fold].copy()
        
        print("Train DF size:", len(train_df))
        debug_fit_and_evaluate(0.5, train_df, train_df)
    else:
        print("Not enough folds!")
