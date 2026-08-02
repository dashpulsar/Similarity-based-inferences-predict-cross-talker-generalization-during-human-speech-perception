import pandas as pd
import numpy as np
import h5py
from joblib import Parallel, delayed

from glmm_prediction.nygaard_glmm import load_h5_data, get_training_talkers_map, create_nygaard_dataset, calculate_nygaard_distances, process_layer_nygaard_l2, fit_and_evaluate_split_nygaard, objective_on_validation_nygaard

import warnings
warnings.filterwarnings('ignore')

EXCEL_PATH = "../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
df_behavioral = pd.read_excel(EXCEL_PATH)

df_test = create_nygaard_dataset(df_behavioral)
print("Test DF shape:", df_test.shape)
print("Folds in test DF:", df_test['fold'].value_counts() if 'fold' in df_test.columns else "NO FOLD COLUMN")

subject_map = get_training_talkers_map(df_behavioral)

h5_path = "../preprocessing/nygaard19_baseline_features.h5"
data_dict = load_h5_data(h5_path)
layer_name = 'MFCC'
layer_data = data_dict[layer_name]

print(f"Layer {layer_name} loaded. Speakers:", list(layer_data.keys()))

dist_df = calculate_nygaard_distances(layer_name, layer_data, df_test, subject_map, distance_type="minkowski", tau=2.0)
print("Dist DF shape:", dist_df.shape)

if not dist_df.empty:
    print(dist_df.head(2))
    res = process_layer_nygaard_l2(layer_name, dist_df, alpha=0.1)
    if res is not None:
        print("GLMM Results:")
        print(res)
    else:
        print("GLMM failed or returned None")
else:
    print("Dist DF is empty!")
