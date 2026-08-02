import h5py
import numpy as np

def inst_norm(X):
    # X shape: (T, 3)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    return (X - mean) / (std + 1e-8)

def apply_inst_norm(input_h5, output_h5):
    print(f"Applying instance norm: {input_h5} -> {output_h5}")
    with h5py.File(input_h5, 'r') as f_in, h5py.File(output_h5, 'w') as f_out:
        for spk in f_in.keys():
            spk_group = f_out.create_group(spk)
            for word in f_in[spk].keys():
                word_group = spk_group.create_group(word)
                for layer in f_in[spk][word].keys():
                    feat = f_in[spk][word][layer][:]
                    feat_norm = inst_norm(feat)
                    word_group.create_dataset(layer, data=feat_norm)
    print(f"Saved to {output_h5}")

apply_inst_norm("nygaard19_tsne_3d_random.h5", "nygaard19_tsne_3d_random_inst_norm.h5")
apply_inst_norm("nygaard19_tsne_3d_ft_random.h5", "nygaard19_tsne_3d_ft_random_inst_norm.h5")
