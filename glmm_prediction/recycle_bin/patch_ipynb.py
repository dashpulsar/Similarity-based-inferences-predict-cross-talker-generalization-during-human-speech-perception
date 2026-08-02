import json

with open('../preprocessing/feature_extraction_nygaard19.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Remove the filter condition
            new_line = line.replace(" and file.lower() in used_filenames_lower", "")
            new_source.append(new_line)
        cell['source'] = new_source

with open('../preprocessing/feature_extraction_nygaard19.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Patched feature_extraction_nygaard19.ipynb successfully!')
