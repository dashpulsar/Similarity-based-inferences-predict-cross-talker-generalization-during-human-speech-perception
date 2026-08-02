import json

notebook_path = '../preprocessing/feature_extraction_nygaard19_baselines.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Comment out the specific lines used for filtering, but we can just replace the if condition
            if "if file.lower().endswith('.wav') and file.lower() in used_filenames_lower:" in line:
                line = line.replace("and file.lower() in used_filenames_lower:", "")
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Patched successfully!')
