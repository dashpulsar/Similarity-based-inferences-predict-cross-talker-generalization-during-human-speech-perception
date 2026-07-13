import json
import re

path = r'c:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\glmm_prediction\xie_glmm.ipynb'
with open(path, 'r', encoding='utf-8') as f: nb = json.load(f)

single_layer_code = ''
for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if 'def process_single_layer_v4' in source:
        single_layer_code = source
        break

# Extract logic
match = re.search(r'(    def run_glmm_logic.*?)    try:\n        layer_data = load_single_layer_from_h5', single_layer_code, re.DOTALL)
glmm_logic_code = match.group(1)

match2 = re.search(r'    try:\n        layer_data = load_single_layer_from_h5[^\n]*\n(.*?)def run_analysis_pipeline_v4', single_layer_code, re.DOTALL)
loop_code = match2.group(1)

# Modify
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    if 'def process_memory_layer' in source:
        new_func = 'def process_memory_layer(layer_key, df_final, audio_dir, layer_data):\n'
        new_func += glmm_logic_code
        new_func += '    try:\n'
        new_func += '        std_data = standardization(layer_data)\n'
        new_func += '        df_dist  = precompute_layer_distances(df_final, audio_dir, std_data, 2, layer_key, func="mean")\n'
        
        loop_code_modified = loop_code.replace('return results_df, layer_key, sim_df', 'if results_df is not None: return results_df')
        new_func += loop_code_modified
        
        source = re.sub(r'def process_memory_layer.*?(?=# =========================================================\n# лȡ GLMM )', new_func, source, flags=re.DOTALL)
        
        # It's possible the header is different due to character encoding, let's just replace the whole function definition
        source = re.sub(r'def process_memory_layer\(layer_key.*?(?=\n# =========================================================)', new_func, source, flags=re.DOTALL)
        
        nb['cells'][i]['source'] = [line + '\n' for line in source.split('\n')]
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Fixed process_memory_layer logic in xie_glmm.ipynb!')
