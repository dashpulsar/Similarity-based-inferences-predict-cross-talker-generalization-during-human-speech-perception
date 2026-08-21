import json

with open("C:/Users/Alex/Documents/GitHub/Similarity-based inferences predict cross-talker generalization during human speech perception/preprocessing/feature_extraction_nygaard19_baselines.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if "mel_padded = F.pad(mel_unsqueeze, (pad_w, pad_w, pad_h, pad_h), mode=\"reflect\")" in line:
                new_source.append("    if mel_unsqueeze.shape[3] > pad_w and mel_unsqueeze.shape[2] > pad_h:\n")
                new_source.append("        mel_padded = F.pad(mel_unsqueeze, (pad_w, pad_w, pad_h, pad_h), mode=\"reflect\")\n")
                new_source.append("    else:\n")
                new_source.append("        mel_padded = F.pad(mel_unsqueeze, (pad_w, pad_w, pad_h, pad_h), mode=\"constant\", value=float(mel_unsqueeze.min()))\n")
            else:
                new_source.append(line)
        cell["source"] = new_source

with open("C:/Users/Alex/Documents/GitHub/Similarity-based inferences predict cross-talker generalization during human speech perception/preprocessing/feature_extraction_nygaard19_baselines.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Patched baseline feature extraction notebook!")
