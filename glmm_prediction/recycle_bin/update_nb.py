import nbformat
import json

nb_path = 'xie_variability_tsne.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Fix build_token_dict bug
for cell in nb.cells:
    if cell.cell_type == 'code' and 'def build_token_dict' in cell.source:
        lines = cell.source.split('\n')
        new_lines = []
        for line in lines:
            if 'tg_sentence[idx_tg - 1].maxTime = tg_sentence[idx_tg].minTime' in line:
                # add if idx_tg > 0 condition
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + 'if idx_tg > 0:')
                new_lines.append(indent + '    tg_sentence[idx_tg - 1].maxTime = tg_sentence[idx_tg].minTime')
            else:
                new_lines.append(line)
        cell.source = '\n'.join(new_lines)

# Insert run_glmm_variability_logic
logic_code = """def run_glmm_variability_logic(tau, train_df, test_df, purpose='optimize'):
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
            ro.r('''
                r_test$SentenceID   <- factor(r_test$SentenceID)
                r_test$Keyword      <- factor(r_test$Keyword)
                r_test$TestTalkerID <- factor(r_test$TestTalkerID)
                model_test <- glmer(
                    cbind(numCorrect, numIncorrect) ~ 1 + variability + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
                    data=r_test, family=binomial(link="logit"), control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=10000)))
                ll_test <- as.numeric(logLik(model_test))
                z_test  <- summary(model_test)$coefficients[2, 3]
            ''')
            return {
                'z_train': z_train, 'z_test': ro.globalenv['z_test'][0],
                'poll_train': ro.globalenv['ll_train'][0] / (train_agg['numCorrect'].sum() + train_agg['numIncorrect'].sum()),
                'poll_test': ro.globalenv['ll_test'][0] / (test_agg['numCorrect'].sum() + test_agg['numIncorrect'].sum()),
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 999.0 if purpose == 'optimize' else None
    return None"""

inserted = False
for cell in nb.cells:
    if cell.cell_type == 'code' and 'def process_single_layer_variability' in cell.source:
        if 'def run_glmm_variability_logic' not in cell.source:
            cell.source = logic_code + '\n\n' + cell.source
            inserted = True
        break

if not inserted:
    print('Failed to insert logic_code or it already exists')

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook updated successfully.')
