import pickle
import h5py

old_pkl_path = r'C:\Users\Alex\Desktop\Pycharm\cross-talker-ASR\July\hubert_Nygaard_all_tr_dict.pkl'
new_h5_path = r'C:\Users\Alex\Documents\GitHub\Similarity-based inferences predict cross-talker generalization during human speech perception\preprocessing\nygaard19_tsne_3d.h5'

with open(old_pkl_path, 'rb') as f:
    old_data = pickle.load(f)

new_data = h5py.File(new_h5_path, 'r')

layer_key_old = 24
layer_key_new = 'tr_24'

old_speakers = list(old_data[layer_key_old].keys())
new_speakers = list(new_data[layer_key_new].keys())

print('Old speakers (first 10):', old_speakers[:10])
print('New speakers (first 10):', new_speakers[:10])

old_spk = old_speakers[0]
new_spk = new_speakers[0]

old_words = list(old_data[layer_key_old][old_spk].keys())
new_words = list(new_data[layer_key_new][new_spk].keys())

print('\nOld words (first 10):', old_words[:10])
print('New words (first 10):', new_words[:10])

new_data.close()
