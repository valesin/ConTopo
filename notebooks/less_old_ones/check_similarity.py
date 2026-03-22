# %%
# Import required libraries
import sys

sys.path.append("..")
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.ensemble_utils import (
    load_similarity_profiles,
    get_ensemble_info,
    iter_ensemble_inference_data_from_config,
)
from utils.experiments import pearson_rdm

# %%
# Select an ensemble by name or hash
ensemble_name = "CE_rho0.2"  # Replace with actual name
profiles = load_similarity_profiles(ensemble_name)
info = get_ensemble_info(ensemble_name)
run_names = info["run_names"]

# %%# %%

# Load embeddings for each model in the ensemble using the ensemble interface
inference_data_list = []
from utils.ensemble_utils import iter_ensemble_inference_data_from_config

ensemble_found = False
for ens in iter_ensemble_inference_data_from_config():
    if ens["ensemble_name"] == ensemble_name:
        for run_name in ens["run_names"]:
            inference_data_list.append(ens["inference_data"][run_name])
        ensemble_found = True
        break
if not ensemble_found:
    raise ValueError(f"Ensemble '{ensemble_name}' not found in config.")

# %%
# Select a test image index
image_idx = 42  # Change as needed
model_embeddings = [model["embeddings"][image_idx] for model in inference_data_list]
similarity_profile = profiles["cosine_results"][
    :, :, image_idx
]  # shape: [num_models, num_classes]
correct_label = inference_data_list[0]["labels"][image_idx]
# %%
# Manual cosine similarity verification
# NOTE: Anchors are now drawn from the CIFAR-10 TRAIN set (not test set),
# and require loading each model's encoder to compute anchor embeddings.
# The old select_anchors() function has been removed.
# To verify correctness, re-run exp_similarityprofiles.py and compare outputs.
print(
    "Manual verification skipped — anchors now come from TRAIN set (see exp_similarityprofiles.py)."
)
# %%
# Visualize similarity profile for the selected image
plt.imshow(similarity_profile.numpy(), aspect="auto", cmap="viridis")
plt.title("Similarity Profile (models x classes)")
plt.xlabel("Class")
plt.ylabel("Model")
plt.colorbar()
plt.show()


# %%
# # Manually calculate RDM for the selected image across models
def calc_triu_rdm(sim_across_models):
    return pearson_rdm(sim_across_models).triu(diagonal=1)


manual_rdm = calc_triu_rdm(similarity_profile)
# Note: The saved key is 'rdm_mats_all' (full 10-class) or 'rdm_mats_remove' (correct-label removed)
print("RDM shape:", manual_rdm.shape)
# %%
