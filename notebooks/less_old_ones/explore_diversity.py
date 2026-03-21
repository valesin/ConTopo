# %%
import torch
div = torch.load("../save/ensembles/0d9e88ef0fce8ae9/pairwise_diversity.pt")
div.keys()
# %%
import numpy as np
pairw = div['q_statistic'].numpy()
upper_vals = pairw[np.triu_indices_from(pairw, k=1)]
upper_vals

# %%
upper_vals.mean()
# %%
all_vals = pairw.flatten()
all_vals.mean()
# %%
import pandas as pd
df = pd.read_csv("../save/ensembles/0d9e88ef0fce8ae9/diversity.csv")
# %%
q_stat = df.loc[df['metric'] == 'q_statistic', 'value'].values[0]
q_stat

# %%
for m in div.keys():
    print(m, div[m].shape, div[m].dtype)
    print(div[m][0, 1])
    print(div[m][1, 0])
# %%
