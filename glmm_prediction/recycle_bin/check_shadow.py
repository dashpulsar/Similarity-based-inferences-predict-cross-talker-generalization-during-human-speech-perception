import pandas as pd
from collections import defaultdict
import os

EXCEL_PATH = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
df = pd.read_excel(EXCEL_PATH)
used_filenames = set(df['FileName'].dropna().unique())
used_filenames_lower = set(f.lower() for f in used_filenames)

AUDIO_DIR = '../data/raw_data/alexander_nygaard19/sound_stimuli'
audio_paths = []
for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.lower().endswith('.wav'):
            audio_paths.append(os.path.join(root, file))

# Group all found files by (speaker, word)
dups = defaultdict(list)
for path in audio_paths:
    meta = os.path.basename(path).lower()
    speaker_id = meta[:3]
    word_name = meta.split(' ')[-1][:-4]
    dups[(speaker_id, word_name)].append(meta)

# Check which ones are in Excel and if they are the first one
shadowed_count = 0
for k, metas in dups.items():
    if len(metas) > 1:
        for i, meta in enumerate(metas):
            if meta in used_filenames_lower:
                if i != 0:
                    print(f"DANGER! The file {meta} is in the Excel file, but it is at index {i}!")
                    print(f"It is shadowed by: {metas[0]}")
                    shadowed_count += 1
                else:
                    print(f"SAFE: The file {meta} is in Excel and is index 0. Shadowing: {metas[1:]}")

print(f"\nTotal shadowed files that were needed: {shadowed_count}")
