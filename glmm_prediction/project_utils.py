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
from numba import njit
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula, pandas2ri
pandas2ri.activate()
warnings.filterwarnings('ignore')

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


@njit(nogil=True)
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


@njit(nogil=True)
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


def process_single_layer_v4(layer_key, df_final, audio_dir, h5_path, tau=2):
    import pandas as pd
    import numpy as np
    import h5py
    import os
    from scipy.optimize import minimize_scalar
    import traceback
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    base   = importr('base')
    stats  = importr('stats')
    lme4   = importr('lme4')
    
    # 支持双向结构的灵活数据加载器
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
            # 自动探测结构: [layer][speaker][sentence] (t-SNE) 或 [speaker][sentence][layer] (原始特征)
            is_layer_first = layer_key in root_keys
            
            for s_idx_mapped, s_idx_original in enumerate(combined_set):
                sent_key = f"sentence_{s_idx_original:02d}"
                for spk in speakers:
                    if is_layer_first:
                        # 对应 xie21_tsne_3d.h5
                        feat = h5f[layer_key][spk][sent_key][:]
                    else:
                        # 对应 xie21_features.h5
                        feat = h5f[spk][sent_key][layer_key][:]
                    layer_data[s_idx_mapped].append(feat)
        return layer_data

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
            ro.r("""
                r_train$SentenceID   <- factor(r_train$SentenceID)
                r_train$Keyword      <- factor(r_train$Keyword)
                r_train$TestTalkerID <- factor(r_train$TestTalkerID)
                model_train <- glmer(
                    cbind(numCorrect, numIncorrect) ~ 1 + similarity + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
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
                test_work['similarity'] = np.exp(-test_work['raw_distance'] * k)

                test_agg = test_work.groupby(['Keyword', 'Condition2', 'TrainingTalkerID1', 'TestTalkerID', 'SentenceID'], as_index=False).agg(
                    IsCorrect=('IsCorrect', 'mean'), similarity=('similarity', 'mean'),
                    numCorrect=('IsCorrect', lambda x: (x == 1).sum()), numIncorrect=('IsCorrect', lambda x: (x == 0).sum())
                )
                ro.globalenv['r_test'] = pandas2ri.py2rpy(test_agg)
                ro.r("""
                    r_test$SentenceID   <- factor(r_test$SentenceID)
                    r_test$Keyword      <- factor(r_test$Keyword)
                    r_test$TestTalkerID <- factor(r_test$TestTalkerID)
                    model_test <- glmer(
                        cbind(numCorrect, numIncorrect) ~ 1 + similarity + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
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

    try:
        layer_data = load_single_layer_from_h5(h5_path, audio_dir, layer_key)
        std_data = standardization(layer_data)
        df_dist  = precompute_layer_distances(df_final, audio_dir, std_data, tau, layer_key, func="mean")
        folds   = sorted(df_dist['fold'].unique())
        diagnostic_results, best_ks = [], []

        for f in folds:
            train_df = df_dist[df_dist['fold'] != f].copy()
            test_df  = df_dist[df_dist['fold'] == f].copy()
            res = minimize_scalar(lambda k: run_glmm_logic(k, train_df, None, 'optimize'), bounds=(0.001, 5.0), method='bounded')
            best_k = res.x
            best_ks.append(best_k)

            metrics = run_glmm_logic(best_k, train_df, test_df, 'evaluate')
            if metrics:
                diagnostic_results.append({
                    'layer': layer_key, 'fold': f, 'type': 'diagnostic', 'k': best_k,
                    'z_train': metrics['z_train'], 'z_test':  metrics['z_test'],
                    'poll_train': metrics['poll_train'], 'poll_test':  metrics['poll_test'],
                    'optimism': (metrics['poll_train'] - metrics['poll_test']) / abs(metrics['poll_train']),
                })

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
                    'poll_train': metrics['poll_train'], 'poll_test':  metrics['poll_test'],
                    'optimism': (metrics['poll_train'] - metrics['poll_test']) / abs(metrics['poll_train']),
                })

        results_df = pd.DataFrame(diagnostic_results + corrected_results)
        sim_df = df_dist[['Keyword', 'TrainingTalkerID', 'TestTalkerID', 'SentenceID', 'raw_distance']].copy()
        sim_df['similarity'] = np.exp(-sim_df['raw_distance'] * mean_k)
        sim_df = sim_df.drop(columns=['raw_distance']).drop_duplicates(subset=['Keyword', 'TrainingTalkerID', 'TestTalkerID', 'SentenceID']).reset_index(drop=True)
        return results_df, layer_key, sim_df

    except Exception as e:
        return f"EXCEPTION in {layer_key}: {e}\n{traceback.format_exc()}"


def run_analysis_pipeline_v4(layers_list, df_final, audio_dir, h5_path, n_jobs=-1):
    print(f"Starting analysis for {len(layers_list)} layers: {layers_list} ...")
    start = time.time()
    
    tasks = [delayed(process_single_layer_v4)(k, df_final, audio_dir, h5_path) for k in layers_list]
    raw_results = Parallel(n_jobs=n_jobs, verbose=5)(tasks)

    valid_results, similarity_frames = [], {}
    for item in raw_results:
        if isinstance(item, str):
            print(item)
            continue
        if item is None: continue
        res_df, layer_key, sim_df = item
        if res_df is not None: valid_results.append(res_df)
        if sim_df is not None: similarity_frames[layer_key] = sim_df

    all_layers_results = pd.concat(valid_results, ignore_index=True) if valid_results else pd.DataFrame()
    df_all = df_final.copy()
    for layer_key, sim_df in similarity_frames.items():
        df_all = df_all.merge(sim_df.rename(columns={'similarity': layer_key}), on=['Keyword', 'TrainingTalkerID', 'TestTalkerID', 'SentenceID'], how='left')

    print(f"Done in {(time.time() - start):.2f}s")
    return all_layers_results, df_all


def optuna_objective(trial, train_df, val_df, alpha):
    k = trial.suggest_float("k", 0.001, 2.0)
    metrics = fit_and_evaluate_split_nygaard(k, train_df, val_df, target_type='val')
    if not metrics['success']:
        return 1000.0
    loss = -metrics['z_target'] + alpha * (k**2)
    return loss


def process_layer_nygaard_l2(layer_key, layer_df, alpha=0.1, n_trials=20):
    try:
        work_df = layer_df.copy()
        rename_map = {'correct': 'Keyword', 'Speaker_full': 'TestTalker', 'accuracy': 'NumCorrect', 'Subject': 'SubjectID'}
        work_df.rename(columns={k:v for k,v in rename_map.items() if k in work_df.columns}, inplace=True)
        if 'NumCorrect' not in work_df.columns: return None
        if 'NumWord' not in work_df.columns: work_df['NumWord'] = 1 
        work_df = work_df.dropna(subset=['raw_distance'])
        if len(work_df) == 0: return None
        
        if 'fold' not in work_df.columns: return None
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
            
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            func = lambda trial: optuna_objective(trial, train_df, val_df, alpha)
            study.optimize(func, n_trials=n_trials)
            
            best_k = study.best_params['k']
            final_metrics = fit_and_evaluate_split_nygaard(best_k, train_df, test_df, target_type='test')
            
            if final_metrics['success']:
                poll_train = final_metrics['loglik_train'] / final_metrics['n_train']
                poll_test = final_metrics['loglik_target'] / final_metrics['n_target']
                opt_metric = (poll_train - poll_test) / abs(poll_train)
                
                results.append({
                    'layer': layer_key, 
                    'fold': test_fold,
                    'type': 'corrected',
                    'alpha': alpha,
                    'k': best_k, 
                    'z_train': final_metrics['z_train'], 
                    'z_test': final_metrics['z_target'],
                    'poll_train': poll_train,
                    'poll_test': poll_test,
                    'optimism': opt_metric
                })
        
        return pd.DataFrame(results)

    except Exception as e:
        print(f"Error in Layer {layer_key}: {e}")
        return pd.DataFrame()


def fit_and_evaluate_split_nygaard(k, train_df, target_df, target_type='val'):
    import os
    os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.4.1'
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.4.1\bin\x64"
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    base = importr('base')
    lme4 = importr('lme4')
    
    try:
        train_work = train_df.copy()
        train_work['similarity'] = np.exp(-train_work['raw_distance'] * k)
        
        groups = ['Keyword', 'TestTalker', 'SubjectID']
        
        train_agg = train_work.groupby(groups, as_index=False).agg(
            similarity=('similarity', 'mean'),
            numCorrect=('NumCorrect', 'sum'),
            numWord=('NumWord', 'sum')
        )
        train_agg['numIncorrect'] = (train_agg['numWord'] - train_agg['numCorrect']).clip(lower=0)
        
        sim_std = train_agg['similarity'].std()
        sim_mean = train_agg['similarity'].mean()
        
        if sim_std < 1e-8:
             return {'success': False, 'reason': 'low_variance'}
             
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
        
        if not res_r.rx2('success')[0]: 
            return {'success': False, 'reason': 'r_error'}

        z_target = res_r.rx2('z_target')[0]
        
        if np.isnan(z_target) or np.isinf(z_target) or abs(z_target) > 30.0:
             return {'success': False, 'reason': 'extreme_z'}

        return {
            'success': True,
            'z_train': res_r.rx2('z_train')[0],
            'loglik_train': res_r.rx2('loglik_train')[0],
            'n_train': train_agg['numWord'].sum(),
            'z_target': z_target,
            'loglik_target': res_r.rx2('loglik_target')[0],
            'n_target': target_agg['numWord'].sum()
        }

    except Exception as e:
        return {'success': False, 'reason': f'exception: {e}'}


@njit(nogil=True)
def weighted_minkowski(vec1, vec2, tau, w=1):
    total = 0.0
    for m in range(len(vec1)):
        diff = w * abs(vec1[m] - vec2[m])
        total += (diff ** tau)
    return total**(1/tau)


def draw_baselines(ax):
    x_limits = [-1, len(order)] 
    ax.axhline(y=mfcc_mean, color='darkorange', linestyle='--', linewidth=2, alpha=0.6)
    ax.fill_between(x_limits, mfcc_mean - mfcc_sem, mfcc_mean + mfcc_sem, color='darkorange', alpha=0.15, zorder=0)
    ax.axhline(y=strf_mean, color='forestgreen', linestyle='-.', linewidth=2, alpha=0.6)
    ax.fill_between(x_limits, strf_mean - strf_sem, strf_mean + strf_sem, color='forestgreen', alpha=0.15, zorder=0)
    ax.axhline(100, color='black', linestyle='-', linewidth=1.0, zorder=0, alpha=0.8)  
    ax.axhline((1.96 / ceil_mean) * 100, color='gray', linestyle=':', linewidth=1.5, zorder=0, alpha=0.6)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.tick_params(axis='x', rotation=45)


def format_label(l):
    if pd.isna(l): return l
    l_str = str(l)
    if l_str == 'MFCC': return 'MFCC'
    if l_str in ['STRFs', 'STRF']: return 'STRF'
    if l_str.startswith('tr_'): return f"Transformer Layer {l_str.split('_')[1]}"
    if l_str.startswith('cnn_'): return f"CNN Layer {l_str.split('_')[1]}"
    return l_str


def get_sort_key(l):
    if l == 'MFCC': return (0, 0)
    if l == 'STRF': return (0, 1)
    if '   ' in l: return (1, 0) 
    type_score = 2 if 'CNN' in l else 3
    nums = re.findall(r'\d+', l)
    num_score = int(nums[0]) if nums else 0
    return (type_score, num_score)


def infer_accent(spk):
    if spk.startswith('e'): return 'English'
    elif spk.startswith('s'): return 'Spanish'
    elif spk.startswith('k'): return 'Korean'
    return 'Other'


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


