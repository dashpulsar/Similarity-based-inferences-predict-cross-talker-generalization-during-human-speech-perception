import nbformat

with open('an19_talker_similarity_T24.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

cell_10 = """# 2. Within vs Between Group Distance Analysis
import scipy.stats as stats
import numpy as np

within_dists = []
between_dists = []

for i in range(len(valid_talkers)):
    for j in range(i+1, len(valid_talkers)):
        t1, t2 = valid_talkers[i], valid_talkers[j]
        g1, g2 = get_talker_group(t1), get_talker_group(t2)
        dist = pairwise_dists.get((t1, t2), np.nan)
        
        if not np.isnan(dist):
            if g1 == g2:
                # Only L1-English, L1-Korean, and L1-Spanish have >1 talker
                within_dists.append(dist)
            else:
                between_dists.append(dist)

mean_within = np.mean(within_dists)
se_within = stats.sem(within_dists)

mean_between = np.mean(between_dists)
se_between = stats.sem(between_dists)

print(f"Mean Within-Group Distance:  {mean_within:.4f} (SE: {se_within:.4f})")
print(f"Mean Between-Group Distance: {mean_between:.4f} (SE: {se_between:.4f})")

# Welch's t-test
t_stat, p_val = stats.ttest_ind(within_dists, between_dists, equal_var=False)
print(f"Welch's t-test: t = {t_stat:.4f}, p = {p_val:.4e}")

# Bootstrap testing
n_boot = 10000
diffs = []
for _ in range(n_boot):
    boot_within = np.random.choice(within_dists, size=len(within_dists), replace=True)
    boot_between = np.random.choice(between_dists, size=len(between_dists), replace=True)
    diffs.append(np.mean(boot_within) - np.mean(boot_between))

p_boot = np.mean(np.array(diffs) >= 0)
print(f"Bootstrap p-value (Within < Between): {p_boot:.4e}")
"""

cell_11 = """# 3. Visualization of Within-Group vs Between-Group Distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for visualization
dist_data = []
for i in range(len(valid_talkers)):
    for j in range(i+1, len(valid_talkers)):
        t1, t2 = valid_talkers[i], valid_talkers[j]
        g1, g2 = get_talker_group(t1), get_talker_group(t2)
        dist = pairwise_dists.get((t1, t2), np.nan)
        
        if not np.isnan(dist):
            is_within = (g1 == g2)
            # Only include groups that have more than 1 talker for 'Within'
            if is_within and g1 not in ['English', 'Korean', 'Spanish']:
                continue
                
            dist_data.append({
                'Talker1': t1,
                'Talker2': t2,
                'Group1': g1,
                'Group2': g2,
                'Distance': dist,
                'Type': 'Within-Group' if is_within else 'Between-Group'
            })

df_dist = pd.DataFrame(dist_data)

plt.figure(figsize=(8, 6))
sns.boxplot(x='Type', y='Distance', data=df_dist, hue='Type', palette='Set2', legend=False)
sns.stripplot(x='Type', y='Distance', data=df_dist, color='black', alpha=0.3, jitter=True)

# Add statistical annotation
y_max = df_dist['Distance'].max()
plt.plot([0, 1], [y_max + 0.02, y_max + 0.02], lw=1.5, c='k')
plt.text(0.5, y_max + 0.025, f"t = {t_stat:.2f}\\n$p = {p_val:.2e}$", ha='center', va='bottom', color='k')

plt.title("Within-Group vs Between-Group Talker Distance (tr_24)", pad=20)
plt.ylabel("Distance (tau=2, k=1)")
plt.ylim(df_dist['Distance'].min() - 0.02, y_max + 0.1)
plt.tight_layout()
plt.show()

# Further breakdown by specific L1 pairs
def get_pair_type(row):
    g1, g2 = sorted([row['Group1'], row['Group2']])
    if row['Type'] == 'Within-Group':
        return f"Within {g1}"
    elif g1 in ['English', 'Korean', 'Spanish'] and g2 in ['English', 'Korean', 'Spanish']:
        return f"Between {g1}-{g2}"
    else:
        return "Between (inc. Int'l)"

df_dist['Pair_Type'] = df_dist.apply(get_pair_type, axis=1)

# Sort order for plotting
order = ['Within English', 'Within Korean', 'Within Spanish', 
         'Between English-Korean', 'Between English-Spanish', 'Between Korean-Spanish', 
         "Between (inc. Int'l)"]

plt.figure(figsize=(12, 6))
sns.boxplot(x='Pair_Type', y='Distance', data=df_dist, order=order, hue='Pair_Type', palette='muted', legend=False)
sns.stripplot(x='Pair_Type', y='Distance', data=df_dist, order=order, color='black', alpha=0.3, jitter=True)
plt.xticks(rotation=45, ha='right')
plt.title("Detailed Breakdown of Pairwise Distances", pad=10)
plt.ylabel("Distance (tau=2, k=1)")
plt.tight_layout()
plt.show()
"""

cell_12 = """# 4. Aggregated Group-Level Distance Heatmap
# Here we calculate the mean distance between each pair of groups and plot it as a heatmap.

# Get all unique groups
all_groups = sorted(list(set([get_talker_group(t) for t in valid_talkers])))

# Initialize the aggregated matrix
agg_dist_matrix = np.zeros((len(all_groups), len(all_groups)))

for i, g1 in enumerate(all_groups):
    for j, g2 in enumerate(all_groups):
        dists = []
        # Find all pairs of talkers between g1 and g2
        talkers1 = [t for t in valid_talkers if get_talker_group(t) == g1]
        talkers2 = [t for t in valid_talkers if get_talker_group(t) == g2]
        
        for t1 in talkers1:
            for t2 in talkers2:
                if t1 != t2: # Exclude self-distance
                    dist = pairwise_dists.get((t1, t2), np.nan)
                    if not np.isnan(dist):
                        dists.append(dist)
                        
        if dists:
            agg_dist_matrix[i, j] = np.mean(dists)
        else:
            agg_dist_matrix[i, j] = np.nan 

plt.figure(figsize=(10, 8))
sns.heatmap(agg_dist_matrix, fmt='.2f', xticklabels=all_groups, yticklabels=all_groups, 
            cmap='inferno_r', annot=True)
plt.title("Aggregated Group-Level Distance (Mean Pairwise)", pad=20)
plt.tight_layout()
plt.show()

# If we just want the main 3 groups for a cleaner condition plot:
main_groups = ['English', 'Korean', 'Spanish']
main_agg_dist_matrix = np.zeros((len(main_groups), len(main_groups)))

for i, g1 in enumerate(main_groups):
    for j, g2 in enumerate(main_groups):
        dists = []
        talkers1 = [t for t in valid_talkers if get_talker_group(t) == g1]
        talkers2 = [t for t in valid_talkers if get_talker_group(t) == g2]
        for t1 in talkers1:
            for t2 in talkers2:
                if t1 != t2:
                    dist = pairwise_dists.get((t1, t2), np.nan)
                    if not np.isnan(dist):
                        dists.append(dist)
        if dists:
            main_agg_dist_matrix[i, j] = np.mean(dists)

plt.figure(figsize=(6, 5))
sns.heatmap(main_agg_dist_matrix, fmt='.2f', xticklabels=main_groups, yticklabels=main_groups, 
            cmap='inferno_r', annot=True)
plt.title("Main Groups (L1-English, L1-Korean, L1-Spanish) \\nAggregated Distance", pad=20)
plt.tight_layout()
plt.show()
"""

cell_13 = """# 5. Separate Talker-to-Talker Heatmaps for English, Korean, and Spanish
groups_to_plot = ['English', 'Korean', 'Spanish']

for g in groups_to_plot:
    # Filter talkers for this group, sorted
    g_talkers = sorted([t for t in valid_talkers if get_talker_group(t) == g])
    
    # Initialize distance matrix
    n = len(g_talkers)
    g_dist_matrix = np.zeros((n, n))
    
    for i, t1 in enumerate(g_talkers):
        for j, t2 in enumerate(g_talkers):
            if i == j:
                g_dist_matrix[i, j] = np.nan
            else:
                g_dist_matrix[i, j] = pairwise_dists.get((t1, t2), 0.0)
                
    mask = np.triu(np.ones_like(g_dist_matrix, dtype=bool))
    
    labels = [get_talker_label(t).replace(f'{g}_', '') for t in g_talkers] 
    
    plt.figure(figsize=(8, 6))
    
    # Use percentiles to clamp colormap for better contrast
    vmin = np.nanpercentile(g_dist_matrix, 5)
    vmax = np.nanpercentile(g_dist_matrix, 95)
    
    sns.heatmap(g_dist_matrix, mask=mask, xticklabels=labels, yticklabels=labels, 
                cmap='inferno_r', annot=True, fmt='.2f', vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Distance'})
    plt.title(f"{g} Group - Talker-to-Talker Distance", pad=20)
    plt.tight_layout()
    plt.show()
"""

nb.cells[10].source = cell_10
nb.cells[11].source = cell_11
nb.cells[12].source = cell_12
nb.cells[13].source = cell_13

with open('an19_talker_similarity_T24.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Cells 10 to 13 have been rewritten cleanly.')
