import pandas as pd
import numpy as np
import h5py
import warnings
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')

from nygaard_glmm import load_h5_data, get_training_talkers_map, create_nygaard_dataset, calculate_nygaard_distances, process_layer_nygaard_l2

if __name__ == "__main__":
    EXCEL_PATH = "../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
    df_behavioral = pd.read_excel(EXCEL_PATH)
    df_test = create_nygaard_dataset(df_behavioral)
    subject_map = get_training_talkers_map(df_behavioral)
    
    h5_path = "../preprocessing/nygaard19_tsne_3d.h5"
    data_dict = load_h5_data(h5_path)
    layer_name = 'cnn_2'
    layer_data = data_dict[layer_name]
    
    dist_df = calculate_nygaard_distances(layer_name, layer_data, df_test, subject_map, distance_type="minkowski", tau=2.0)
    print("Dist DF shape:", dist_df.shape)
    
    res = process_layer_nygaard_l2(layer_name, dist_df, alpha=0.1)
    print("Process Result:")
    print(res)
