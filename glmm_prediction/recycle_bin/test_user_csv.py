import pandas as pd
from nygaard_glmm import run_full_nygaard_analysis

print("Loading data...")
df_behavioral = pd.read_excel('../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx')
df_raw = pd.read_csv('C:/Users/Alex/Desktop/Pycharm/cross-talker-ASR/cross-validation/Nygaard_TR_CNN_Raw_Distance_Global_1.csv')
df_tr2 = df_raw[df_raw['layer'] == 'tr_2']

print("Running GLMM...")
res = run_full_nygaard_analysis(df_behavioral, {'tr_2': df_tr2}, 0.1)
print(res)
