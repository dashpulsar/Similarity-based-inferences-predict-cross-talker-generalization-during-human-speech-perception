import nbformat as nbf

notebook_path = 'an19_talker_similarity_T24.ipynb'
nb = nbf.read(notebook_path, as_version=4)

code_cell_4 = nbf.v4.new_code_cell("""
# 4. Aggregated Group-Level Correlation/Similarity Heatmap
# Here we calculate the mean similarity between each pair of groups and plot it as a heatmap.

# Get all unique groups
all_groups = sorted(list(set([get_talker_group(t) for t in valid_talkers])))

# Initialize the aggregated matrix
agg_sim_matrix = np.zeros((len(all_groups), len(all_groups)))

for i, g1 in enumerate(all_groups):
    for j, g2 in enumerate(all_groups):
        sims = []
        # Find all pairs of talkers between g1 and g2
        talkers1 = [t for t in valid_talkers if get_talker_group(t) == g1]
        talkers2 = [t for t in valid_talkers if get_talker_group(t) == g2]
        
        for t1 in talkers1:
            for t2 in talkers2:
                if t1 != t2: # Exclude self-similarity
                    # The dictionary might have (t1, t2) or (t2, t1)
                    sim = pairwise_sims.get((t1, t2), np.nan)
                    if not np.isnan(sim):
                        sims.append(sim)
                        
        if sims:
            agg_sim_matrix[i, j] = np.mean(sims)
        else:
            agg_sim_matrix[i, j] = np.nan # e.g., Within-group for a group with only 1 talker

plt.figure(figsize=(10, 8))
sns.heatmap(agg_sim_matrix, xticklabels=all_groups, yticklabels=all_groups, 
            cmap='inferno', annot=True, fmt='.3f', vmin=np.nanmin(agg_sim_matrix), vmax=np.nanmax(agg_sim_matrix))
plt.title("Aggregated Group-Level Similarity (Mean Pairwise)", pad=20)
plt.tight_layout()
plt.show()

# If we just want the main 3 groups for a cleaner condition plot:
main_groups = ['English', 'Korean', 'Spanish']
main_agg_matrix = np.zeros((len(main_groups), len(main_groups)))

for i, g1 in enumerate(main_groups):
    for j, g2 in enumerate(main_groups):
        sims = []
        talkers1 = [t for t in valid_talkers if get_talker_group(t) == g1]
        talkers2 = [t for t in valid_talkers if get_talker_group(t) == g2]
        for t1 in talkers1:
            for t2 in talkers2:
                if t1 != t2:
                    sim = pairwise_sims.get((t1, t2), np.nan)
                    if not np.isnan(sim):
                        sims.append(sim)
        if sims:
            main_agg_matrix[i, j] = np.mean(sims)

plt.figure(figsize=(6, 5))
sns.heatmap(main_agg_matrix, xticklabels=main_groups, yticklabels=main_groups, 
            cmap='inferno', annot=True, fmt='.3f')
plt.title("Main Groups (L1-English, L1-Korean, L1-Spanish) \\nAggregated Similarity", pad=20)
plt.tight_layout()
plt.show()
""")

nb.cells.append(code_cell_4)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Aggregated heatmap cell added to the notebook.")
