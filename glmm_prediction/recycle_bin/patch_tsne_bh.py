import json

with open('../preprocessing/feature_extraction_nygaard19.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The last cell is the openTSNE cell
last_cell = nb['cells'][-1]
new_source = []
for line in last_cell['source']:
    if 'tsne = TSNE(n_components=TSNE_DIM, random_state=42, n_jobs=1)' in line:
        line = line.replace('tsne = TSNE(n_components=TSNE_DIM, random_state=42, n_jobs=1)', 
                            'tsne = TSNE(n_components=TSNE_DIM, random_state=42, n_jobs=1, negative_gradient_method=\"bh\")')
    new_source.append(line)

nb['cells'][-1]['source'] = new_source

with open('../preprocessing/feature_extraction_nygaard19.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Patched successfully!')
