with open('../preprocessing/feature_extraction_nygaard19_baselines.ipynb', 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace('mode="reflect"', 'mode="constant", value=0')
with open('../preprocessing/feature_extraction_nygaard19_baselines.ipynb', 'w', encoding='utf-8') as f:
    f.write(c)
