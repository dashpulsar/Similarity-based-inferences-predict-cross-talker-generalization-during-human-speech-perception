import nbformat
import json

nb_path = 'xie_variability_tsne.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Replace run_glmm_variability_logic with a version that has pandas2ri.activate()
logic_code = """def run_glmm_variability_logic(tau, train_df, test_df, purpose='optimize'):
    try:
        from rpy2.robjects import pandas2ri
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        lme4 = importr('lme4')
        
        train_work = train_df.copy()
        if 'TrainingTalkerID1' not in train_work.columns:
            train_work['TrainingTalkerID1'] = train_work['TrainingTalkerID'].apply(lambda x: ",".join(sorted(str(x).split(", "))))
        
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
                test_work['TrainingTalkerID1'] = test_work['TrainingTalkerID'].apply(lambda x: ",".join(sorted(str(x).split(", "))))

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

for cell in nb.cells:
    if cell.cell_type == 'code' and 'def run_glmm_variability_logic' in cell.source:
        # replace the cell source with the logic_code followed by the rest of the cell, without duplicating the old logic_code
        # But wait, earlier I just prepended logic_code to the cell with process_single_layer_variability.
        # Let's split by 'def process_single_layer_variability'
        parts = cell.source.split('def process_single_layer_variability')
        if len(parts) > 1:
            cell.source = logic_code + '\n\ndef process_single_layer_variability' + parts[1]
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook updated successfully.')
