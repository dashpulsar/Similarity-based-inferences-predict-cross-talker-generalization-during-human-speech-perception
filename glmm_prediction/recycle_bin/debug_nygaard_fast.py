import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from nygaard_glmm import load_h5_data, calculate_nygaard_distances, process_layer_nygaard_l2, create_nygaard_dataset, get_training_talkers_map

print("Loading behavioral data...")
EXCEL_PATH = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
df_behavioral = pd.read_excel(EXCEL_PATH)
df_test = create_nygaard_dataset(df_behavioral)
subject_map = get_training_talkers_map(df_behavioral)

print("Loading one layer features...")
h5_path = '../preprocessing/nygaard19_tsne_3d.h5'
data_dict = load_h5_data(h5_path)

layer_name = 'cnn_2'
layer_data = data_dict[layer_name]

print("Calculating distance...")
df_dist = calculate_nygaard_distances(layer_name, layer_data, df_test, subject_map, "minkowski", 2.0)

if df_dist.empty:
    print("df_dist is empty!")
else:
    print(f"df_dist calculated, shape: {df_dist.shape}")
    print("Running process_layer_nygaard_l2...")
    try:
        # Step-by-step logic of process_layer_nygaard_l2
        work_df = df_dist.copy()
        rename_map = {'correct': 'Keyword', 'Speaker_full': 'TestTalker', 'accuracy': 'NumCorrect', 'Subject': 'SubjectID'}
        work_df.rename(columns={k:v for k,v in rename_map.items() if k in work_df.columns}, inplace=True)
        print("Columns after rename:", work_df.columns)
        
        if 'NumCorrect' not in work_df.columns:
            print("ERROR: NumCorrect is not in work_df.columns!")
            
        if 'NumWord' not in work_df.columns: 
            work_df['NumWord'] = 1 
            
        work_df = work_df.dropna(subset=['raw_distance'])
        if len(work_df) == 0: 
            print("ERROR: work_df is empty after dropna!")
            
        if 'fold' not in work_df.columns: 
            print("ERROR: fold not in work_df.columns!")
            
        work_df['fold'] = work_df['fold'].astype(int)
        folds = sorted(work_df['fold'].unique())
        num_folds = len(folds)
        if num_folds < 3: 
            print(f"ERROR: num_folds < 3 (is {num_folds})!")
            
        print("All checks passed. Calling process_layer_nygaard_l2 directly...")
        df_res = process_layer_nygaard_l2(layer_name, df_dist, n_trials=2)
        print(f"Result from process_layer_nygaard_l2 shape: {df_res.shape}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

