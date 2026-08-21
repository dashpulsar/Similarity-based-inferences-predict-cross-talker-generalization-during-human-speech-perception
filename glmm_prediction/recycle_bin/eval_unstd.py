import pandas as pd
import json

# Import the necessary functions directly from nygaard_glmm.ipynb context
# Wait, it's easier to just copy the baseline code from nygaard_glmm.ipynb
# Let's run it by importing nygaard_glmm functions if possible, or we can just parse the notebook and execute it
with open('nygaard_glmm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
code_text = ""
for cell in code_cells:
    code_text += "".join(cell) + "\n"

# We can execute the definition parts to get `run_full_nygaard_analysis`
exec_globals = {}
# Filter out the execution part
exec_code = []
for line in code_text.split('\n'):
    if line.startswith('if os.path.exists(EXCEL_PATH):'):
        break
    exec_code.append(line)

exec('\n'.join(exec_code), exec_globals)

run_full_nygaard_analysis = exec_globals['run_full_nygaard_analysis']

EXCEL_PATH = "../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx"
df_behavioral = pd.read_excel(EXCEL_PATH)

print("Running baseline WITHOUT standardization...")
df_baseline_unstd = run_full_nygaard_analysis(
    df_behavioral, 
    "nygaard19_baseline_features_unstd.h5", 
    distance_type="minkowski", 
    tau=2.0, 
    alpha=0.1, 
    n_jobs=30
)

df_baseline_unstd.to_csv('nygaard19_glmm_results_baseline_unstd.csv', index=False)
print("Finished evaluating unstandardized baseline!")

# Now print the average Z-score for MFCC and STRF
z_mfcc = df_baseline_unstd[df_baseline_unstd['layer'] == 'MFCC']['normalized_z'].mean()
z_strf = df_baseline_unstd[df_baseline_unstd['layer'] == 'STRF']['normalized_z'].mean()

print(f"MFCC Unstandardized Mean Normalized Z: {z_mfcc:.2f}%")
print(f"STRF Unstandardized Mean Normalized Z: {z_strf:.2f}%")
