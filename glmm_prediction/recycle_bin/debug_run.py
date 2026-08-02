import pandas as pd
from nygaard_glmm import run_full_nygaard_analysis

EXCEL_PATH = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
df_behavioral = pd.read_excel(EXCEL_PATH)

print("Starting debug run...")
df_tsne_3d = run_full_nygaard_analysis(df_behavioral, '../preprocessing/nygaard19_tsne_3d.h5', distance_type='minkowski', tau=2.0, alpha=0.1, n_jobs=30)
print(df_tsne_3d)
