import nbformat

with open('an19_talker_similarity_T24.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Cell 3 updates
        cell.source = cell.source.replace('total_sim = 0.0', 'total_dist = 0.0')
        cell.source = cell.source.replace('sim = np.exp(-dist * 1.0) \\n        total_sim += sim', 'total_dist += dist')
        cell.source = cell.source.replace('sim = np.exp(-dist * 1.0)\\n        total_sim += sim', 'total_dist += dist')
        cell.source = cell.source.replace('return t1, t2, total_sim / count', 'return t1, t2, total_dist / count')
        cell.source = cell.source.replace('for t1, t2, avg_sim in results:', 'for t1, t2, avg_dist in results:')
        cell.source = cell.source.replace('if avg_sim is not None:', 'if avg_dist is not None:')
        cell.source = cell.source.replace('pairwise_sims[(t1, t2)] = avg_sim', 'pairwise_dists[(t1, t2)] = avg_dist')
        cell.source = cell.source.replace('pairwise_sims[(t2, t1)] = avg_sim', 'pairwise_dists[(t2, t1)] = avg_dist')
        cell.source = cell.source.replace('pairwise_sims = {}', 'pairwise_dists = {}')
        cell.source = cell.source.replace('avg_sim', 'avg_dist')
        cell.source = cell.source.replace('total_sim', 'total_dist')
        
        # Cell 4 updates
        cell.source = cell.source.replace('sim = pairwise_sims.get((t1, t2), 0.0)', 'dist = pairwise_dists.get((t1, t2), 0.0)')
        cell.source = cell.source.replace('dist_matrix[i, j] = 1.0 - sim # Distance for clustering', 'dist_matrix[i, j] = dist # Distance for clustering')
        cell.source = cell.source.replace('sim_matrix[i, j] = pairwise_sims.get((t1, t2), 0.0)', 'sim_matrix[i, j] = pairwise_dists.get((t1, t2), 0.0)')
        cell.source = cell.source.replace('sim_matrix = np.zeros', 'dist_matrix_plot = np.zeros')
        cell.source = cell.source.replace('sim_matrix[i, j] = np.nan', 'dist_matrix_plot[i, j] = np.nan')
        cell.source = cell.source.replace('sim_matrix[i, j] = pairwise_dists.get((t1, t2), 0.0)', 'dist_matrix_plot[i, j] = pairwise_dists.get((t1, t2), 0.0)')
        
        # New cells updates
        cell.source = cell.source.replace('new_sim_matrix', 'new_dist_matrix')
        cell.source = cell.source.replace('pairwise_sims', 'pairwise_dists')
        cell.source = cell.source.replace('sim = pairwise_dists', 'dist = pairwise_dists')
        cell.source = cell.source.replace('sims = []', 'dists = []')
        cell.source = cell.source.replace('sims.append', 'dists.append')
        cell.source = cell.source.replace('if sims:', 'if dists:')
        cell.source = cell.source.replace('np.mean(sims)', 'np.mean(dists)')
        
        cell.source = cell.source.replace('agg_sim_matrix', 'agg_dist_matrix')
        cell.source = cell.source.replace('main_agg_matrix', 'main_agg_dist_matrix')
        cell.source = cell.source.replace('g_sim_matrix', 'g_dist_matrix')
        
        cell.source = cell.source.replace('within_sims', 'within_dists')
        cell.source = cell.source.replace('between_sims', 'between_dists')
        cell.source = cell.source.replace('sim_data', 'dist_data')
        cell.source = cell.source.replace('df_sim', 'df_dist')
        
        cell.source = cell.source.replace('np.isnan(sim)', 'np.isnan(dist)')
        
        # Labels and Titles
        cell.source = cell.source.replace('Similarity', 'Distance')
        cell.source = cell.source.replace('similarity', 'distance')
        cell.source = cell.source.replace('sim = ', 'dist = ')
        cell.source = cell.source.replace('Within > Between', 'Within < Between')
        cell.source = cell.source.replace('<= 0', '>= 0')
        
        # Revert x 10000 scaling and format
        cell.source = cell.source.replace('* 10000', '')
        cell.source = cell.source.replace('x 10,000', '')
        cell.source = cell.source.replace("fmt='.1f'", "fmt='.2f'")
        
        # Reverse colormap so smaller distance is brighter
        cell.source = cell.source.replace("cmap='inferno'", "cmap='inferno_r'")
        
with open('an19_talker_similarity_T24.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Notebook updated to use Distance.')
