import pandas as pd
import numpy as np
from nygaard_glmm import load_h5_data, calculate_nygaard_distances, create_nygaard_dataset, get_training_talkers_map

print("Loading behavioral data...")
EXCEL_PATH = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
df_behavioral = pd.read_excel(EXCEL_PATH)
df_test = create_nygaard_dataset(df_behavioral)
subject_map = get_training_talkers_map(df_behavioral)

print("Loading one layer features...")
h5_path = '../preprocessing/nygaard19_tsne_3d.h5'
data_dict = load_h5_data(h5_path)

layer_name = 'tr_2'
if layer_name not in data_dict:
    # tr_2 might be named differently
    for k in data_dict.keys():
        if 'tr' in k and '2' in k:
            layer_name = k
            break

layer_data = data_dict[layer_name]

print("Calculating distance...")
df_dist = calculate_nygaard_distances(layer_name, layer_data, df_test, subject_map, "minkowski", 2.0)

print(f"My mean raw_distance for {layer_name} is: {df_dist['raw_distance'].mean()}")
