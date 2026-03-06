#%%
import os
import json
import matplotlib.pyplot as plt

# Add parent directory to path to import utils
import sys
sys.path.append(os.path.abspath('..'))

from utils.ensemble_utils import list_ensembles, get_ensemble_path_by_name, get_ensemble_info

#%%
# 1. Collect results from all available ensembles
results = []

for ens_info in list_ensembles():
    name = ens_info["name"]
    # We are only interested in CE_rho* ensembles
    if not name or not name.startswith("CE_rho"):
        continue
        
    # Extract the rho coefficient
    try:
        rho_str = name.replace("CE_rho", "")
        rho = float(rho_str)
    except ValueError:
        print(f"Skipping {name}, could not parse rho coefficient")
        continue
        
    # Locate the metrics JSON file
    ensemble_dir = get_ensemble_path_by_name(name)
    metrics_path = os.path.join(ensemble_dir, "adapters_metrics.json")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        accs = metrics.get("final_test_accuracies", {})
        acc_embed = accs.get("acc_embed")
        acc_embed_sim = accs.get("acc_embed_sim")
        
        if acc_embed is not None and acc_embed_sim is not None:
            # Get the mean accuracy and ensemble accuracy from get_ensemble_info
            try:
                info = get_ensemble_info(name)
                comp_acc = info.get("comp_mean_acc", 0.0)
                ensemble_acc = info.get("ensemble_accs", {}).get("soft", 0.0)
            except Exception as e:
                print(f"Failed to load extra info for {name}: {e}")
                comp_acc = 0.0
                ensemble_acc = 0.0
                
            results.append({
                "name": name,
                "rho": rho,
                "acc_embed": acc_embed,
                "acc_embed_sim": acc_embed_sim,
                "comp_acc": comp_acc,
                "ensemble_acc": ensemble_acc
            })
            
print(f"Found complete adapter metrics for {len(results)} ensembles.")

#%%
# 2. Extract and organize data for plotting
# Sort by rho coefficient so the plot lines are continuous
results.sort(key=lambda x: x["rho"])

rhos = [r["rho"] for r in results]
acc_embeds = [r["acc_embed"] for r in results]
acc_embed_sims = [r["acc_embed_sim"] for r in results]
comp_accs = [r["comp_acc"] for r in results]
ensemble_accs = [r["ensemble_acc"] for r in results]

#%%
# 3. Plot the accuracies across rhos
plt.figure(figsize=(10, 6))

# Plot lines with distinct colors and markers
# Create an evenly spaced x-axis for categorical plotting
x_positions = list(range(len(rhos)))

plt.plot(x_positions, acc_embeds, marker='o', markersize=8, linestyle='-', linewidth=2.5, 
         color='dodgerblue', label='Features: Embeddings Only')

plt.plot(x_positions, acc_embed_sims, marker='s', markersize=8, linestyle='-', linewidth=2.5, 
         color='crimson', label='Features: Embeddings + Similarity Profile')
         
plt.plot(x_positions, ensemble_accs, marker='^', markersize=8, linestyle='--', linewidth=2.0, 
         color='forestgreen', label='Ensemble (Soft Voting)')
         
plt.plot(x_positions, comp_accs, marker='x', markersize=8, linestyle=':', linewidth=2.0, 
         color='gray', label='Mean Component Accuracy')

# Aesthetics
plt.title('Adapter Test Accuracy vs. Rho', fontsize=16, pad=15)
plt.xlabel('Rho', fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=14)

# Set the x-ticks to the categorical positions and label them
plt.xticks(ticks=x_positions, labels=[str(r) for r in rhos], fontsize=12)
plt.yticks(fontsize=12)

# Remove symlog/log scaling since we are doing categorical spacing
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()

# Save the plot explicitly just in case we need to view it outside the notebook
plt.savefig('adapters_accuracy_vs_rho.png', dpi=300)
plt.show()

#%%
# 4. Print a tabulated summary
print(f"{'Ensemble':<15} | {'Rho':<8} | {'Comp Mean':<10} | {'Ens Soft':<10} | {'Embed Only':<11} | {'Embed+Sim':<11} | {'Diff'}")
print("-" * 85)
for r in results:
    diff = r['acc_embed_sim'] - r['acc_embed']
    print(f"{r['name']:<15} | {r['rho']:<8.3f} | {r['comp_acc']:<10.4f} | {r['ensemble_acc']:<10.4f} | {r['acc_embed']:<11.4f} | {r['acc_embed_sim']:<11.4f} | {diff:+.4f}")
