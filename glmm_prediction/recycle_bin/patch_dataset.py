import json

notebook_path = '../preprocessing/feature_extraction_nygaard19_baselines.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'grp_mfcc[speaker_id].create_dataset(' in line:
                indent = line.split('grp_mfcc')[0]
                new_source.append(indent + "if word_name not in grp_mfcc[speaker_id]:\n")
                new_source.append(indent + "    " + line.strip() + "\n")
                continue
            if 'grp_strf[speaker_id].create_dataset(' in line:
                indent = line.split('grp_strf')[0]
                new_source.append(indent + "if word_name not in grp_strf[speaker_id]:\n")
                new_source.append(indent + "    " + line.strip() + "\n")
                continue
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Patched dataset creation successfully!')
