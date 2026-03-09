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
        acc_linear_embed = accs.get("acc_linear_embed")
        acc_linear_embed_sim = accs.get("acc_linear_embed_sim")

        if all(x is not None for x in [acc_embed, acc_embed_sim, acc_linear_embed, acc_linear_embed_sim]):
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
                "acc_linear_embed": acc_linear_embed,
                "acc_linear_embed_sim": acc_linear_embed_sim,
                "comp_acc": comp_acc,
                "ensemble_acc": ensemble_acc
            })
            
print(f"Found complete adapter metrics for {len(results)} ensembles.")

#%%
# 2. Extract and organize data for plotting
# Sort by rho coefficient so the plot lines are continuous
rhos = [r["rho"] for r in results]
results.sort(key=lambda x: x["rho"])

rhos = [r["rho"] for r in results]
acc_embeds = [r["acc_embed"] for r in results]
acc_embed_sims = [r["acc_embed_sim"] for r in results]
acc_linear_embeds = [r["acc_linear_embed"] for r in results]
acc_linear_embed_sims = [r["acc_linear_embed_sim"] for r in results]
comp_accs = [r["comp_acc"] for r in results]
ensemble_accs = [r["ensemble_acc"] for r in results]

#%%
# 3. Plot the accuracies across rhos
plt.figure(figsize=(10, 6))
x_positions = list(range(len(rhos)))

plt.plot(x_positions, acc_linear_embeds, marker='o', markersize=8, linestyle='-', linewidth=2.5,
         color='navy', label='Linear (Embed Only)')
plt.plot(x_positions, acc_embeds, marker='o', markersize=8, linestyle='--', linewidth=2.5,
         color='dodgerblue', label='MLP (Embed Only)')

plt.plot(x_positions, acc_linear_embed_sims, marker='s', markersize=8, linestyle='-', linewidth=2.5,
         color='darkred', label='Linear (Embed+Sim)')
plt.plot(x_positions, acc_embed_sims, marker='s', markersize=8, linestyle='--', linewidth=2.5,
         color='crimson', label='MLP (Embed+Sim)')

plt.plot(x_positions, ensemble_accs, marker='^', markersize=8, linestyle='--', linewidth=2.0,
         color='forestgreen', label='Ensemble (Soft Voting)')
plt.plot(x_positions, comp_accs, marker='x', markersize=8, linestyle=':', linewidth=2.0,
         color='gray', label='Mean Component Accuracy')

plt.title('Adapter Test Accuracy vs. Rho', fontsize=16, pad=15)
plt.xlabel('Rho', fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=14)
plt.xticks(ticks=x_positions, labels=[str(r) for r in rhos], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig('adapters_accuracy_vs_rho.png', dpi=300)
plt.show()

#%%
# 4. Print a tabulated summary
print(f"{'Ensemble':<15} | {'Rho':<8} | {'Comp Mean':<10} | {'Ens Soft':<10} | {'Lin Embed':<10} | {'MLP Embed':<10} | {'Lin Emb+Sim':<12} | {'MLP Emb+Sim':<12}")
print("-" * 110)
for r in results:
    print(f"{r['name']:<15} | {r['rho']:<8.3f} | {r['comp_acc']:<10.4f} | {r['ensemble_acc']:<10.4f} | "
          f"{r['acc_linear_embed']:<10.4f} | {r['acc_embed']:<10.4f} | "
          f"{r['acc_linear_embed_sim']:<12.4f} | {r['acc_embed_sim']:<12.4f}")

#%%
# 5. Print a LaTeX-formatted table
print("\n" + "="*85)
print("LaTeX Table Format:")
print("="*85 + "\n")

print("\\begin{table}[htpb!]")
print("  \\centering")
print("  \\caption{Adapter performance evaluated on the testing split (\\texttt{ensemble\\_test}), comparing representations formed by concatenated target embeddings with versus without similarity (RDM) features constraint. The ensemble size is $K=10$ and the number of similarity features is $D_{\\text{sim}}=45$.}")
print("  \\label{tab:adapter_results}")
print("  \\begin{tabular}{lcccc}")
print("    \\hline")
print("    Ensemble & Mean Comp. & Soft Voting & Acc (Embed Only) & Acc (Embed + Sim) \\\\")
print("    \\hline")

for r in results:
    # We can infer K from D_sim if we knew it, or just pull M if it was saved.
    # But since the JSON parsing above didn't explicitly extract M/D_sim into `results`,
    # let's reload them briefly for complete LaTeX printing if missing,
    # or just assume K=5 and D_sim=10 for CIFAR-10 rho experiments if standard. 
    # Let's cleanly pull it from the original JSON to make it robust!
    metrics_path = os.path.join(get_ensemble_path_by_name(r['name']), "adapters_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        K = metrics.get('M', '?')
        D_sim = metrics.get('D_sim', '?')
        
    # Formatting accuracies to percentages (e.g. 0.9452 -> 94.52%)
    acc_only = r['acc_embed'] * 100
    acc_sim = r['acc_embed_sim'] * 100
    comp_acc = r['comp_acc'] * 100
    ens_acc = r['ensemble_acc'] * 100
    
    # Escape underscores in ensemble names for LaTeX
    safe_name = r['name'].replace('_', '\\_')
    
    print(f"    \\texttt{{{safe_name}}} & {comp_acc:.2f}\\% & {ens_acc:.2f}\\% & {acc_only:.2f}\\% & {acc_sim:.2f}\\% \\\\")

print("    \\hline")
print("  \\end{tabular}")
print("\\end{table}")
