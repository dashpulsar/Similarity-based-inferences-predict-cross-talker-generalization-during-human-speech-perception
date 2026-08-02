import nbformat

nb_path = 'xie_variability_tsne.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_code = """def _minkowski_distance(x, y, tau):
    diff = np.abs(x - y)
    return np.sum(diff ** tau, axis=-1) ** (1.0 / tau)

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

def compute_variability_for_condition(training_talkers, sent_range, tokens_for_unit, talker_index_map, tau, calculation_type):
    if calculation_type == 'WithinToken':
        token_variances = []
        for each_talker in training_talkers:
            talker_idx = find_talker_idx(each_talker, talker_index_map)
            if talker_idx == -1: continue
            for sent_idx in sent_range:
                for token_frames in tokens_for_unit[talker_idx][sent_idx]:
                    if token_frames.shape[0] == 0: continue
                    token_mean = np.mean(token_frames, axis=0)
                    frame_dists = _minkowski_distance(token_frames, token_mean, tau)
                    token_variances.append(np.mean(frame_dists))
        return np.mean(token_variances) if token_variances else np.nan

    elif calculation_type == 'WithinType':
        type_variances = []
        for sent_idx in sent_range:
            max_tokens = 0
            for each_talker in training_talkers:
                talker_idx = find_talker_idx(each_talker, talker_index_map)
                if talker_idx == -1: continue
                max_tokens = max(max_tokens, len(tokens_for_unit[talker_idx][sent_idx]))
            
            for token_pos in range(max_tokens):
                token_means = []
                for each_talker in training_talkers:
                    talker_idx = find_talker_idx(each_talker, talker_index_map)
                    if talker_idx == -1: continue
                    if token_pos < len(tokens_for_unit[talker_idx][sent_idx]):
                        frames = tokens_for_unit[talker_idx][sent_idx][token_pos]
                        if frames.shape[0] > 0:
                            token_means.append(np.mean(frames, axis=0))
                if token_means:
                    type_mean = np.mean(token_means, axis=0)
                    token_dists = [_minkowski_distance(tm, type_mean, tau) for tm in token_means]
                    type_variances.append(np.mean(token_dists))
        return np.mean(type_variances) if type_variances else np.nan

    elif calculation_type == 'BetweenType':
        type_means = []
        for sent_idx in sent_range:
            max_tokens = 0
            for each_talker in training_talkers:
                talker_idx = find_talker_idx(each_talker, talker_index_map)
                if talker_idx == -1: continue
                max_tokens = max(max_tokens, len(tokens_for_unit[talker_idx][sent_idx]))
            
            for token_pos in range(max_tokens):
                token_means = []
                for each_talker in training_talkers:
                    talker_idx = find_talker_idx(each_talker, talker_index_map)
                    if talker_idx == -1: continue
                    if token_pos < len(tokens_for_unit[talker_idx][sent_idx]):
                        frames = tokens_for_unit[talker_idx][sent_idx][token_pos]
                        if frames.shape[0] > 0:
                            token_means.append(np.mean(frames, axis=0))
                if token_means:
                    type_means.append(np.mean(token_means, axis=0))
        if not type_means: return np.nan
        global_mean = np.mean(type_means, axis=0)
        type_dists = [_minkowski_distance(tm, global_mean, tau) for tm in type_means]
        return np.mean(type_dists)

    elif calculation_type == 'Order':
        token_variances = []
        for each_talker in training_talkers:
            talker_idx = find_talker_idx(each_talker, talker_index_map)
            if talker_idx == -1: continue
            for sent_idx in sent_range:
                for token_frames in tokens_for_unit[talker_idx][sent_idx]:
                    if token_frames.shape[0] < 2: continue
                    dists = _minkowski_distance(token_frames[1:], token_frames[:-1], tau)
                    token_variances.append(np.mean(dists))
        return np.mean(token_variances) if token_variances else np.nan
"""

# Replace the cell containing def _generalized_variance
found = False
for cell in nb.cells:
    if cell.cell_type == 'code' and 'def _generalized_variance' in cell.source:
        cell.source = new_code
        found = True
        break

if not found:
    print('Failed to find _generalized_variance cell')
else:
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('Notebook mathematically updated successfully.')
