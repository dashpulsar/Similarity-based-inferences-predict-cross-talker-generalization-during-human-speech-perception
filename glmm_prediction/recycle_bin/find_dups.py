import os

AUDIO_DIR = '../data/raw_data/alexander_nygaard19/sound_stimuli'
audio_paths = []
for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.lower().endswith('.wav'):
            audio_paths.append(os.path.join(root, file))

seen = set()
duplicates = []
for path in audio_paths:
    meta = os.path.basename(path)
    speaker_id = meta[:3].lower()
    word_name = meta.lower().split(' ')[-1][:-4]
    pair = (speaker_id, word_name)
    if pair in seen:
        duplicates.append(meta)
    seen.add(pair)

print('Number of duplicates:', len(duplicates))
if len(duplicates) > 0:
    print('First 10 duplicates:', duplicates[:10])
