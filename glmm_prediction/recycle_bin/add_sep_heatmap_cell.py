import nbformat as nbf

notebook_path = 'an19_talker_similarity_T24.ipynb'
nb = nbf.read(notebook_path, as_version=4)

code_cell_5 = nbf.v4.new_code_cell("""
# 5. Separate Talker-to-Talker Heatmaps for English, Korean, and Spanish
groups_to_plot = ['English', 'Korean', 'Spanish']

for g in groups_to_plot:
    # Filter talkers for this group, sorted
    g_talkers = sorted([t for t in valid_talkers if get_talker_group(t) == g])
    
    # Initialize similarity matrix
    n = len(g_talkers)
    g_sim_matrix = np.zeros((n, n))
    
    for i, t1 in enumerate(g_talkers):
        for j, t2 in enumerate(g_talkers):
            if i == j:
                g_sim_matrix[i, j] = np.nan
            else:
                g_sim_matrix[i, j] = pairwise_sims.get((t1, t2), 0.0)
                
    # Scale by 10,000 for readability
    g_sim_matrix = g_sim_matrix * 10000
    
    mask = np.triu(np.ones_like(g_sim_matrix, dtype=bool))
    
    labels = [get_talker_label(t).replace(f'{g}_', '') for t in g_talkers] # Simplify label by removing redundant prefix
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(g_sim_matrix, mask=mask, xticklabels=labels, yticklabels=labels, 
                cmap='inferno', annot=True, fmt='.1f')
    plt.title(f"{g} Group - Talker-to-Talker Similarity (x 10,000)", pad=20)
    plt.tight_layout()
    plt.show()
""")

nb.cells.append(code_cell_5)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Separate heatmaps cell added to the notebook.")
