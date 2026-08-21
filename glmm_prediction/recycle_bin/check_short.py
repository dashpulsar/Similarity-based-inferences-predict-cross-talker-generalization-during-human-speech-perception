import os
import torchaudio

AUDIO_DIR = '../data/raw_data/alexander_nygaard19/sound_stimuli'
audio_paths = []
for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.lower().endswith('.wav'):
            audio_paths.append(os.path.join(root, file))

min_len = 99999999
min_path = ''
for path in audio_paths:
    try:
        info = torchaudio.info(path)
        if info.num_frames < min_len:
            min_len = info.num_frames
            min_path = path
    except Exception as e:
        print(f"Error reading {path}: {e}")

print('Shortest audio:', min_path, 'frames:', min_len)
