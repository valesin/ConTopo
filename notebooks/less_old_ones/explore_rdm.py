# %%
import torch
import sys

sys.path.append("..")
from utils.experiments import pearson_corrcoef

rdm = torch.load("../save/ResNet18/models/CE_rho0.04/RDM_CE_rho0.04.pt")
rdm.keys()

# %%
upper = rdm["rdms_upper"]
(upper, len(upper))
# %%
import polars as pl

df = pl.from_dict({str(i): t.tolist() for i, t in enumerate(upper)})
corr = df.corr()
# %%
import numpy as np

corr_np = corr.to_numpy()
upper_vals = corr_np[np.triu_indices_from(corr_np, k=1)]
mean = upper_vals.mean()
mean

# %%
cons = torch.load("../save/ResNet18/models/CE_rho0.04/RDMConsistency_CE_rho0.04.pt")
cons
# %%
import itertools

comb = list(itertools.combinations(range(10), 2))
comb

# %%
display(corr)
print(comb[0], [comb[0][0], comb[0][1]])
corr.select([str(comb[0][0]), str(comb[0][1])])
# %%
for c in comb:
    # f = corr.select([str(c[0]), str(c[1])])
    v = corr[c[0], [c[1]]]
    print(v)
    t1 = upper[c[0]]
    t2 = upper[c[1]]
    a = t1.flatten()
    b = t2.flatten()
    corr2 = pearson_corrcoef(torch.stack([a, b]))[0, 1]
    print("Correlation:", corr2.item())

    # display(f)
