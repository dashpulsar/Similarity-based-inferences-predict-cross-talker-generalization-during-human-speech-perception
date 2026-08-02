import json

notebook_path = '../preprocessing/feature_extraction_nygaard19_baselines.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if line.strip() == "if file.lower().endswith('.wav')":
                line = line.replace("if file.lower().endswith('.wav')", "if file.lower().endswith('.wav'):")
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Patched missing colon successfully!')
