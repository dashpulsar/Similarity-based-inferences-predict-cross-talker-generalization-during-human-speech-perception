import json

path = r'c:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\glmm_prediction\test_nygaard_visualization.ipynb'
extract_path = r'c:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\preprocessing\extract_inst_norm.py'
eval_path = r'c:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\glmm_prediction\eval_inst_norm.py'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(extract_path, 'r', encoding='utf-8') as f:
    extract_code = f.read()
    
with open(eval_path, 'r', encoding='utf-8') as f:
    eval_code = f.read()

new_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### Appendix: Instance-Normalized Baseline Extraction & Evaluation\n',
        'The following cells contain the logic that extracts MFCC and STRF features with **Instance Normalization** (z-scoring per audio file), and the evaluation logic that generated `nygaard19_glmm_results_baseline_inst_norm.csv`.'
    ]
}

new_code_extract = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [line + '\n' for line in extract_code.split('\n')]
}

new_code_eval = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [line + '\n' for line in eval_code.split('\n')]
}

nb['cells'].extend([new_md, new_code_extract, new_code_eval])

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Appended instance normalization code to the notebook.')
