import pandas as pd
from collections import defaultdict

EXCEL_PATH = '../data/raw_data/alexander_nygaard19/AN19-exposure-test-behavioral-data.xlsx'
df = pd.read_excel(EXCEL_PATH)
used_filenames = df['FileName'].dropna().unique()

dups_in_excel = defaultdict(list)
for filename in used_filenames:
    speaker_id = filename[:3].lower()
    word_name = filename.lower().split(' ')[-1][:-4]
    dups_in_excel[(speaker_id, word_name)].append(filename)

count = 0
for k, v in dups_in_excel.items():
    if len(v) > 1:
        count += 1
        print(f"Collision in Excel for key {k}: {v}")

print(f"\nTotal collisions in Excel: {count}")
