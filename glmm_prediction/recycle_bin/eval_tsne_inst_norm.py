import pandas as pd
import json
import h5py
import os

with open('nygaard_glmm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
code_text = ""
for cell in code_cells:
    code_text += "".join(cell) + "\n"

exec_globals = {}
exec_code = []
for line in code_text.split('\n'):
    if line.startswith('if os.path.exists(EXCEL_PATH):'):
        break
    exec_code.append(line)

exec('\n'.join(exec_code), exec_globals)

run_full_nygaard_analysis = exec_globals['run_full_nygaard_analysis']

EXCEL_PATH = "../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
df_behavioral = pd.read_excel(EXCEL_PATH)

print("Evaluating Inst-Norm TSNE Features...")
df_tsne_inst = run_full_nygaard_analysis(
    df_behavioral, 
    "../preprocessing/nygaard19_tsne_3d_random_inst_norm.h5",
    distance_type="minkowski", 
    tau=2.0, 
    alpha=0.1, 
    n_jobs=30
)

if not df_tsne_inst.empty:
    df_tsne_inst.to_csv('nygaard19_glmm_results_hubert_inst_norm.csv', index=False)
    print("Saved nygaard19_glmm_results_hubert_inst_norm.csv")

print("Evaluating Inst-Norm FT TSNE Features...")
df_tsne_ft_inst = run_full_nygaard_analysis(
    df_behavioral, 
    "../preprocessing/nygaard19_tsne_3d_ft_random_inst_norm.h5",
    distance_type="minkowski", 
    tau=2.0, 
    alpha=0.1, 
    n_jobs=30
)

if not df_tsne_ft_inst.empty:
    df_tsne_ft_inst.to_csv('nygaard19_glmm_results_hubert_ft_inst_norm.csv', index=False)
    print("Saved nygaard19_glmm_results_hubert_ft_inst_norm.csv")

print("Evaluation Complete!")
