import pandas as pd
import numpy as np
import os
import h5py
import traceback
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.1'
os.environ['PATH'] += os.pathsep + r'C:\Program Files\R\R-4.4.1\bin\x64'
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

def load_single_layer_from_h5(h5_path, audio_dir, layer_key):
    def get_pathset(paths):
        return [os.path.join(dir, each_file) for dir, mid, files in os.walk(paths) for each_file in files if each_file.endswith('.wav')]
    audio_paths = get_pathset(audio_dir)[::-1]
    speakers = [os.path.basename(p).replace('.wav', '') for p in audio_paths]
    set1_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]
    set2_list = [17,18,19,20,21,22,24,25,26,27,28,29,30,31,37,40]
    combined_set = set1_list + set2_list
    
    layer_data = [ [] for _ in range(32) ]
    with h5py.File(h5_path, 'r') as h5f:
        root_keys = list(h5f.keys())
        is_layer_first = layer_key in root_keys
        
        for s_idx_mapped, s_idx_original in enumerate(combined_set):
            sent_key = f'sentence_{s_idx_original:02d}'
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

def _generalized_variance(diff, tau):
    if diff.ndim == 1:
        return np.sum(np.abs(diff) ** tau)
    return np.sum(np.abs(diff) ** tau, axis=1)

def get_training_paths(TrainingTalkerID):
    TalkerID = []
    if not isinstance(TrainingTalkerID, str): return []
    for each_ID in TrainingTalkerID.split(', '):
        if each_ID[:3] == 'CMN':
            TalkerID.append(f'ALL_{each_ID[-3:]}_M_CMN')
        else:
            TalkerID.append(f'ALL_{each_ID[-3:]}_M_ENG')
    return TalkerID

def find_talker_idx(talker, talker_index_map):
    for spk, idx in talker_index_map.items():
        if spk.startswith(talker):
            return idx
    return -1

def compute_variability_for_condition(training_talkers, sent_range, reduced_data, talker_index_map, tau, method):
    if method == 'BetweenSentence':
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
    return np.nan

def precompute_layer_variability(df, reduced_data, speakers, tau, method):
    talker_index_map = {spk: idx for idx, spk in enumerate(speakers)}
    unique_conditions = df[['TrainingTalkerID', 'TrainingTestSet']].drop_duplicates()
    var_map = {}
    for row in unique_conditions.itertuples(index=False):
        tr_id = row.TrainingTalkerID
        tr_set = str(row.TrainingTestSet)
        if pd.isna(tr_id): continue
        
        training_talkers = get_training_paths(tr_id)
        if tr_set.startswith('set1'):
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

def run_glmm_variability_logic(tau, train_df, test_df, purpose='optimize'):
    try:
        train_work = train_df.copy()
        if 'TrainingTalkerID1' not in train_work.columns:
            train_work['TrainingTalkerID1'] = train_work['TrainingTalkerID'].apply(lambda x: ','.join(sorted(x.split(', '))))
        
        train_agg = train_work.groupby(
            ['Keyword', 'Condition2', 'TrainingTalkerID1', 'TestTalkerID', 'SentenceID'], as_index=False
        ).agg(
            IsCorrect=('IsCorrect', 'mean'), variability=('variability', 'mean'),
            numCorrect=('IsCorrect', lambda x: (x == 1).sum()), numIncorrect=('IsCorrect', lambda x: (x == 0).sum())
        )
        train_agg = train_agg.dropna(subset=['variability'])
        
        train_sd = train_agg['variability'].std()
        if train_sd == 0 or pd.isna(train_sd): 
            print('TRAIN SD IS 0 OR NAN', train_sd)
            return 999.0 if purpose == 'optimize' else None

        ro.globalenv['r_train'] = pandas2ri.py2rpy(train_agg)
        ro.r('''
            r_train$SentenceID   <- factor(r_train$SentenceID)
            r_train$Keyword      <- factor(r_train$Keyword)
            r_train$TestTalkerID <- factor(r_train$TestTalkerID)
            model_train <- glmer(
                cbind(numCorrect, numIncorrect) ~ 1 + variability + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
                data=r_train, family=binomial(link="logit"), control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=10000)))
            z_train  <- summary(model_train)$coefficients[2, 3]
            ll_train <- as.numeric(logLik(model_train))
        ''')
        z_train = ro.globalenv['z_train'][0]
        if purpose == 'optimize': return -z_train

    except Exception as e:
        print(f"EXCEPTION inside glmm: {e}")
        print(traceback.format_exc())
        return 999.0 if purpose == 'optimize' else None
    return None


audio_dir = r'../data/raw_data/xie_liu_jaeger21/sound_stimuli'
human_result_path = r'../data/raw_data/xie_liu_jaeger21/X21-exposure-test-behavioral-data.xlsx'
h5_path = r'../preprocessing/xie21_tsne_3d_ft.h5'

human_result = pd.read_excel(human_result_path)
human_result_1a = human_result[human_result['Experiment']=='1a'].copy()
human_result_1a['TrainingTalkerID1'] = human_result_1a['TrainingTalkerID'].astype(str).apply(lambda x: ','.join(sorted(x.split(', '))) if pd.notna(x) else x)

df = human_result_1a.copy()
participants = df[['WorkerID', 'TrainingTestSet', 'Condition2', 'TestTalkerID']].drop_duplicates().reset_index(drop=True)
participants['combined_key'] = participants['TrainingTestSet'].astype(str) + '_' + participants['Condition2'].astype(str) + '_' + participants['TestTalkerID'].astype(str)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
participants['fold'] = -1
for fold_idx, (train_index, test_index) in enumerate(skf.split(participants, participants['combined_key'])):
    participants.loc[test_index, 'fold'] = fold_idx + 1
df_final = df.merge(participants[['WorkerID', 'fold']], on='WorkerID', how='left')

layer_key = 'cnn_6'
layer_data, speakers = load_single_layer_from_h5(h5_path, audio_dir, layer_key)
std_data = standardization(layer_data)

train_df = df_final[df_final['fold'] != 1].copy()
df_var = precompute_layer_variability(train_df, std_data, speakers, 2.0, 'BetweenSentence')

run_glmm_variability_logic(2.0, df_var, None, 'optimize')
