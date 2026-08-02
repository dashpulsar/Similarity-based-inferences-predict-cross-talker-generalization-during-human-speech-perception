import nbformat as nbf

notebook_path = 'an19_talker_similarity_T24.ipynb'
nb = nbf.read(notebook_path, as_version=4)

code_cell_3 = nbf.v4.new_code_cell("""
# 3. Visualization of Within-Group vs Between-Group Similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for visualization
sim_data = []
for i in range(len(valid_talkers)):
    for j in range(i+1, len(valid_talkers)):
        t1, t2 = valid_talkers[i], valid_talkers[j]
        g1, g2 = get_talker_group(t1), get_talker_group(t2)
        sim = pairwise_sims.get((t1, t2), np.nan)
        
        if not np.isnan(sim):
            is_within = (g1 == g2)
            # Only include groups that have more than 1 talker for 'Within'
            if is_within and g1 not in ['English', 'Korean', 'Spanish']:
                continue
                
            sim_data.append({
                'Talker1': t1,
                'Talker2': t2,
                'Group1': g1,
                'Group2': g2,
                'Similarity': sim,
                'Type': 'Within-Group' if is_within else 'Between-Group'
            })

df_sim = pd.DataFrame(sim_data)

plt.figure(figsize=(8, 6))
sns.boxplot(x='Type', y='Similarity', data=df_sim, palette='Set2')
sns.stripplot(x='Type', y='Similarity', data=df_sim, color='black', alpha=0.3, jitter=True)

# Add statistical annotation
y_max = df_sim['Similarity'].max()
plt.plot([0, 1], [y_max + 0.02, y_max + 0.02], lw=1.5, c='k')
plt.text(0.5, y_max + 0.025, f"t = {t_stat:.2f}\\n$p = {p_val:.2e}$", ha='center', va='bottom', color='k')

plt.title("Within-Group vs Between-Group Talker Similarity (tr_24)", pad=20)
plt.ylabel("Similarity (tau=2, k=1)")
plt.ylim(df_sim['Similarity'].min() - 0.02, y_max + 0.1)
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

df_sim['Pair_Type'] = df_sim.apply(get_pair_type, axis=1)

# Sort order for plotting
order = ['Within English', 'Within Korean', 'Within Spanish', 
         'Between English-Korean', 'Between English-Spanish', 'Between Korean-Spanish', 
         "Between (inc. Int'l)"]

plt.figure(figsize=(12, 6))
sns.boxplot(x='Pair_Type', y='Similarity', data=df_sim, order=order, palette='muted')
sns.stripplot(x='Pair_Type', y='Similarity', data=df_sim, order=order, color='black', alpha=0.3, jitter=True)
plt.xticks(rotation=45, ha='right')
plt.title("Detailed Breakdown of Pairwise Similarities", pad=10)
plt.ylabel("Similarity (tau=2, k=1)")
plt.tight_layout()
plt.show()
""")

nb.cells.extend([code_cell_3])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Visualization cell added to the notebook.")
