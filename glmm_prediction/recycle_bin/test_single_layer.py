import pandas as pd
from nygaard_glmm import run_full_nygaard_analysis

if __name__ == '__main__':
    df_behavioral = pd.read_excel('../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx')
    h5_file = '../preprocessing/nygaard19_tsne_3d.h5'
    layers = ['tr_10']
    
    print("Running layer tr_10...")
    df_res = run_full_nygaard_analysis(df_behavioral, h5_file, "minkowski", 2.0, layers, n_jobs=1)
    print("Done. Result:")
    print(df_res)
