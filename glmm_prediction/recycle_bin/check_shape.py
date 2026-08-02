import pickle
import sys
with open(r'C:\Users\Alex\Desktop\Pycharm\cross-talker-ASR\cross-validation\hubert_Nygaard_all_tr_dict.pkl', 'rb') as f:
    d = pickle.load(f)
    print("Keys in top level:", list(d.keys())[:5])
    # The dictionary structure was: { layer_id: {speaker: {word: vector}} } based on the notebook
    # Wait, the notebook did:
    # for k, v in reduced_dict.items():
    #     new_key = f"tr_{k}"
    # So the top level is layer (0, 2, 4...)
    layer_key = list(d.keys())[0]
    layer_dict = d[layer_key]
    first_speaker = list(layer_dict.keys())[0]
    first_word = list(layer_dict[first_speaker].keys())[0]
    feat = layer_dict[first_speaker][first_word]
    print('Shape of feature:', feat.shape)
