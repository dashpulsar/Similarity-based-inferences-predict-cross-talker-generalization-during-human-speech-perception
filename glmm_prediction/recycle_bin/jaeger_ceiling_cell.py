# ============================================================================
# Behavioral Noise Ceiling: Jaeger Self-Predictability Method
# ============================================================================
#
# Florian Jaeger's specification:
#   "It is the self predictability of the data.
#    So glmm(cbind(n_incorrect, n_correct) ~ logodds_correct
#    for current item during training.
#    Fit to training and then used during test to derive predictions,
#    just like for the models we are comparing."
#
# Key principles:
#   1. Predictor = log-odds(correct) per item, from TRAINING data only
#   2. Same CV & GLMM evaluation pipeline as model comparisons
#   3. No data leakage: test fold information never enters the predictor
# ============================================================================

import sys, os, time

if '.' not in sys.path: sys.path.append('.')
if '..' not in sys.path: sys.path.append('..')
try:
    from nygaard_glmm import create_nygaard_dataset
except ImportError:
    sys.path.append('glmm_prediction')
    from nygaard_glmm import create_nygaard_dataset

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# ── Load raw behavioral data ──
print('Loading raw behavioral dataset for ceiling computation...')
excel_path = 'data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
if not os.path.exists(excel_path):
    excel_path = '../' + excel_path
df_raw = pd.read_excel(excel_path)
df_test_ceil = create_nygaard_dataset(df_raw)
print(f'  {len(df_test_ceil)} test trials, '
      f'{df_test_ceil["Subject"].nunique()} subjects, '
      f'{df_test_ceil["correct"].nunique()} keywords, '
      f'{df_test_ceil["Speaker_full"].nunique()} talkers, '
      f'{df_test_ceil["fold"].nunique()} folds')


def compute_jaeger_ceiling(df_data, item_groups=['correct', 'Speaker_full']):
    """
    Compute behavioral noise ceiling via Jaeger's self-predictability method.

    For each CV fold:
      1. From TRAINING data only, compute per-item log-odds(correct)
      2. Map training-derived log-odds to test items as predictor
      3. Scale predictor using training-fold statistics (Gelman, 2008)
      4. Fit GLMM on test data with training-derived predictor
      5. Extract Wald z-statistic

    Parameters
    ----------
    df_data : DataFrame with 'correct', 'Speaker_full', 'accuracy', 'Subject', 'fold'
    item_groups : columns defining a unique item (default: word x talker)

    Returns
    -------
    np.array of Wald z-statistics, one per CV fold
    """
    lme4 = importr('lme4')
    folds = sorted(df_data['fold'].unique())
    z_vals = []

    for test_fold in folds:
        t0 = time.time()
        print(f'\nFold {test_fold} (test):')

        # ── Step 1: Split data ──
        # All non-test folds serve as training.
        # No validation fold is needed because the predictor (log-odds)
        # has no tunable hyperparameter (unlike model similarities which tune k).
        train_df = df_data[df_data['fold'] != test_fold].copy()
        test_df  = df_data[df_data['fold'] == test_fold].copy()
        print(f'  Train: {len(train_df)} trials from '
              f'{train_df["Subject"].nunique()} subjects')
        print(f'  Test:  {len(test_df)} trials from '
              f'{test_df["Subject"].nunique()} subjects')

        # ── Step 2: Compute per-item log-odds from TRAINING data only ──
        # "item" = unique combination of item_groups (default: word x talker)
        item_stats = train_df.groupby(item_groups).agg(
            n_correct=('accuracy', 'sum'),
            n_total=('accuracy', 'count')
        ).reset_index()
        item_stats['p_correct'] = (
            item_stats['n_correct'] / item_stats['n_total']
        ).clip(0.01, 0.99)  # clip to avoid log(0) or log(inf)
        item_stats['logodds_correct'] = np.log(
            item_stats['p_correct'] / (1 - item_stats['p_correct'])
        )
        n_items = len(item_stats)
        lo_min = item_stats['logodds_correct'].min()
        lo_max = item_stats['logodds_correct'].max()
        print(f'  Items: {n_items} (logodds range: [{lo_min:.2f}, {lo_max:.2f}])')

        # ── Step 3: Map training-derived log-odds to train AND test ──
        # Test items receive the logodds computed from training subjects only.
        # This is the key difference from the old approach, which computed
        # the predictor from ALL folds (data leakage).
        merge_cols = item_groups + ['logodds_correct']
        train_merged = train_df.merge(
            item_stats[merge_cols], on=item_groups, how='left'
        ).dropna(subset=['logodds_correct'])
        test_merged = test_df.merge(
            item_stats[merge_cols], on=item_groups, how='left'
        ).dropna(subset=['logodds_correct'])

        if len(test_merged) == 0:
            print('  ERROR: No overlapping items between train and test.')
            continue

        # ── Step 4: Scale using TRAINING statistics ──
        # Following Gelman (2008): scale by (x - mean) / (2 * SD)
        lo_mean = train_merged['logodds_correct'].mean()
        lo_std  = train_merged['logodds_correct'].std()
        if lo_std == 0:
            print('  ERROR: Zero variance in log-odds.')
            continue

        train_merged['logodds_scaled'] = (
            train_merged['logodds_correct'] - lo_mean
        ) / (2 * lo_std)
        test_merged['logodds_scaled'] = (
            test_merged['logodds_correct'] - lo_mean
        ) / (2 * lo_std)

        # ── Step 5: Aggregate by Subject x Item for GLMM ──
        glmm_groups = ['Subject'] + item_groups
        test_agg = test_merged.groupby(glmm_groups, as_index=False).agg(
            logodds_scaled=('logodds_scaled', 'first'),
            numCorrect=('accuracy', 'sum'),
            numWord=('accuracy', 'count')
        )
        test_agg['numIncorrect'] = (
            test_agg['numWord'] - test_agg['numCorrect']
        ).clip(lower=0)
        test_agg.rename(columns={
            'correct': 'Keyword',
            'Speaker_full': 'TestTalker',
            'Subject': 'SubjectID'
        }, inplace=True)

        # ── Step 6: Fit GLMM on test data via R ──
        # The predictor (logodds_scaled) is derived entirely from training.
        # We fit the GLMM on test data to evaluate whether this predictor
        # significantly explains test-fold accuracy - paralleling exactly
        # how model comparisons are evaluated.
        ro.globalenv['r_test'] = pandas2ri.py2rpy(test_agg)
        try:
            ro.r('''
                library(lme4)
                r_test$Keyword   <- factor(r_test$Keyword)
                r_test$TestTalker <- factor(r_test$TestTalker)
                r_test$SubjectID <- factor(r_test$SubjectID)

                model_test <- tryCatch({
                    glmer(cbind(numCorrect, numIncorrect) ~ logodds_scaled
                              + (1 + logodds_scaled | SubjectID),
                          data = r_test, family = binomial(link = "logit"),
                          control = glmerControl(optimizer = "bobyqa",
                                                 optCtrl = list(maxfun = 1e5)))
                }, error = function(e) { NULL })

                ceil_res <- list()
                if (!is.null(model_test)) {
                    ceil_res$z_test <- summary(model_test)$coefficients[2, 3]
                    ceil_res$success <- TRUE
                } else {
                    ceil_res$success <- FALSE
                }
                ceil_res
            ''')

            res_r = ro.globalenv['ceil_res']
            if res_r.rx2('success')[0]:
                z_t = res_r.rx2('z_test')[0]
                z_vals.append(z_t)
                print(f'  z_test = {z_t:.4f} ({time.time()-t0:.1f}s)')
            else:
                print(f'  GLMM fitting failed.')
        except Exception as e:
            print(f'  R error: {e}')

    return np.array(z_vals)


# ── Run the ceiling computation ──
print('\n' + '='*60)
print('Computing Jaeger Self-Predictability Ceiling')
print('  Item: word x talker (correct x Speaker_full)')
print('  Predictor: log-odds(correct) from training fold only')
print('='*60)

ceiling_vals = compute_jaeger_ceiling(df_test_ceil)

ceil_mean = np.mean(ceiling_vals)
ceil_sem  = np.std(ceiling_vals, ddof=1) / np.sqrt(len(ceiling_vals))

print('\n' + '='*60)
print('=== Jaeger Self-Predictability Ceiling ===')
print(f'  Per-fold z: {ceiling_vals}')
print(f'  Mean z:     {ceil_mean:.4f} (SEM: {ceil_sem:.4f})')
print('='*60)

# ── Set variables for downstream visualization cells ──
loso_ceiling_vals = ceiling_vals
loso_mean = ceil_mean
loso_sem = ceil_sem

df_all['percent_ceiling'] = (df_all['z_test'] / ceil_mean) * 100.0
ceil_sem_pc = (ceil_sem / ceil_mean) * 100.0
df_all['percent_ceiling_loso'] = df_all['percent_ceiling'].copy()
loso_sem_pc = ceil_sem_pc

print(f'\nPrimary Benchmark: 100.0% (z = {ceil_mean:.2f})')
print(f'Confidence Band: 100.0% +/- {ceil_sem_pc:.2f}%')
