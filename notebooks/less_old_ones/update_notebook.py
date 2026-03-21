import json

notebook_path = "/home/vlr/Workspaces/Topographic/ConTopo/notebooks/diversity_consistency_correlation_all_rhos.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        modified = False
        for line in cell['source']:
            if line.strip().startswith("rhos = [5, 1, 0.2, 0.04, 0.008, 0]"):
                new_source.append("rhos_tuples = [('high', 5), ('med', 1), ('low', 0.2), ('very low', 0.04), ('lowest', 0.008), ('zero', 0)]\n")
                new_source.append("rhos = [r[1] for r in rhos_tuples]\n")
                modified = True
            else:
                new_source.append(line)
        if modified:
            cell['source'] = new_source
            break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
