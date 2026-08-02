import os
from collections import defaultdict
AUDIO_DIR = '../data/raw_data/alexander_nygaard19/sound_stimuli'
audio_paths = []
for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.lower().endswith('.wav'):
            audio_paths.append(os.path.join(root, file))

dups = defaultdict(list)
for path in audio_paths:
    meta = os.path.basename(path)
    speaker_id = meta[:3].lower()
    word_name = meta.lower().split(' ')[-1][:-4]
    dups[(speaker_id, word_name)].append(path)

count = 0
print('Duplicate file pairs:')
for k, v in dups.items():
    if len(v) > 1:
        count += 1
        if count <= 10:
            print(f'\nPair {count}:')
            for p in v:
                print('  ' + p)

print(f'\nTotal {count} groups of duplicate (speaker, word) mappings found.')
