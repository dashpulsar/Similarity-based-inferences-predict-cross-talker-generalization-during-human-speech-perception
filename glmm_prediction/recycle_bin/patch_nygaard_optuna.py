import json

filepath = "C:/Users/Alex/Documents/GitHub/Similarity-based inferences predict cross-talker generalization during human speech perception/glmm_prediction/nygaard_glmm.ipynb"

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell that contains `process_layer_nygaard_l2` and replace it entirely with Optuna code.
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def process_layer_nygaard_l2" in source or "def objective_on_validation_nygaard" in source:
            # We found the cell with the old logic. Let's replace the whole source for these functions.
            new_source = """
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def fit_and_evaluate_split_nygaard(k, train_df, target_df, target_type='val'):
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
                glmer(cbind(numCorrect, numIncorrect) ~ 1 + similarity_scaled + (1|TestTalker) + (1|Keyword), 
                      data=r_train, family=binomial(link="logit"), 
                      control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
            }, error=function(e){ NULL })
            
            model_target <- tryCatch({
                glmer(cbind(numCorrect, numIncorrect) ~ 1 + similarity_scaled + (1|TestTalker) + (1|Keyword), 
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
        return {'success': False, 'reason': 'exception'}

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
"""
            cell["source"] = [line + "\n" for line in new_source.split("\n")]

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Nygaard Optuna script patch applied.")
