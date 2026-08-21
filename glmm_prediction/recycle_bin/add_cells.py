import nbformat as nbf

notebook_path = 'an19_talker_similarity_T24.ipynb'
nb = nbf.read(notebook_path, as_version=4)

md_cell = nbf.v4.new_markdown_cell("""
## Updated Visualizations & Analysis (per Advisor's Comments)
Here we redo the plots and statistical comparisons:
1. Show only the lower triangle (no diagonal).
2. Treat international talkers as their specific language backgrounds (1 talker per language per gender).
3. Label talkers by their language.
4. Calculate within vs between group similarity means and SE, followed by a Welch's t-test and bootstrap analysis.
""")

code_cell_1 = nbf.v4.new_code_cell("""
# 1. Language mapping for all talkers
l1_map = {
    'if1': 'Albanian',
    'if2': 'Dutch',
    'if3': 'French',
    'if4': 'German',
    'if5': 'Japanese',
    'if6': 'Somali',
    'im1': 'Romanian',
    'im2': 'Bengali',
    'im3': 'Russian',
    'im4': 'Chinese',
    'im5': 'Turkish',
    'im6': 'Indian'
}

def get_talker_label(t):
    if t.startswith('e'):
        return f"English_{t}"
    elif t.startswith('k'):
        return f"Korean_{t}"
    elif t.startswith('s'):
        return f"Spanish_{t}"
    elif t in l1_map:
        return f"{l1_map[t]}_{t}"
    return t
    
def get_talker_group(t):
    if t.startswith('e'): return 'English'
    if t.startswith('k'): return 'Korean'
    if t.startswith('s'): return 'Spanish'
    return l1_map.get(t, 'Other')

# We can re-sort the talkers so males/females are grouped together cleanly
english_talkers = sorted([t for t in valid_talkers if get_talker_group(t) == 'English'])
spanish_talkers = sorted([t for t in valid_talkers if get_talker_group(t) == 'Spanish'])
korean_talkers = sorted([t for t in valid_talkers if get_talker_group(t) == 'Korean'])
intl_talkers = sorted([t for t in valid_talkers if t.startswith('i')])

# Concatenate all sorted talkers
new_ordered_talkers = english_talkers + spanish_talkers + korean_talkers + intl_talkers
ordered_labels = [get_talker_label(t) for t in new_ordered_talkers]

# Rebuild the similarity matrix with the new ordering
n_total = len(new_ordered_talkers)
new_sim_matrix = np.zeros((n_total, n_total))

for i, t1 in enumerate(new_ordered_talkers):
    for j, t2 in enumerate(new_ordered_talkers):
        if i == j:
            new_sim_matrix[i, j] = np.nan # Use NaN so it doesn't skew the heatmap color scale
        else:
            new_sim_matrix[i, j] = pairwise_sims.get((t1, t2), 0.0)

# Create lower triangle mask
mask = np.triu(np.ones_like(new_sim_matrix, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(new_sim_matrix, mask=mask, xticklabels=ordered_labels, yticklabels=ordered_labels, 
            cmap='inferno', annot=False, fmt='.2f', vmin=np.nanmin(new_sim_matrix), vmax=np.nanmax(new_sim_matrix))
plt.title(f"Talker Pairwise Similarity (tau=2, k=1) - {layer_name}\\n(Lower Triangle)")
plt.tight_layout()
plt.show()
""")

code_cell_2 = nbf.v4.new_code_cell("""
# 2. Within vs Between Group Similarity Analysis
import scipy.stats as stats

within_sims = []
between_sims = []

for i in range(len(valid_talkers)):
    for j in range(i+1, len(valid_talkers)):
        t1, t2 = valid_talkers[i], valid_talkers[j]
        g1, g2 = get_talker_group(t1), get_talker_group(t2)
        sim = pairwise_sims.get((t1, t2), np.nan)
        
        if not np.isnan(sim):
            if g1 == g2:
                # Only L1-English, L1-Korean, and L1-Spanish have >1 talker
                within_sims.append(sim)
            else:
                between_sims.append(sim)

mean_within = np.mean(within_sims)
se_within = stats.sem(within_sims)

mean_between = np.mean(between_sims)
se_between = stats.sem(between_sims)

print(f"Mean Within-Group Similarity:  {mean_within:.4f} (SE: {se_within:.4f})")
print(f"Mean Between-Group Similarity: {mean_between:.4f} (SE: {se_between:.4f})")

# Welch's t-test
t_stat, p_val = stats.ttest_ind(within_sims, between_sims, equal_var=False)
print(f"Welch's t-test: t = {t_stat:.4f}, p = {p_val:.4e}")

# Bootstrap testing
n_boot = 10000
diffs = []
for _ in range(n_boot):
    boot_within = np.random.choice(within_sims, size=len(within_sims), replace=True)
    boot_between = np.random.choice(between_sims, size=len(between_sims), replace=True)
    diffs.append(np.mean(boot_within) - np.mean(boot_between))

p_boot = np.mean(np.array(diffs) <= 0)
print(f"Bootstrap p-value (Within > Between): {p_boot:.4e}")
""")

nb.cells.extend([md_cell, code_cell_1, code_cell_2])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Added new cells to the notebook.")
