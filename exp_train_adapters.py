#%% Imports + configuration path parsing + standardization flag
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timezone
from torch.utils.data import DataLoader, TensorDataset

import utils.ensemble_utils as ensemble_utils
from utils import env

# Configuration
APPLY_STANDARDIZATION = True
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We iterate models according to `run_names` order. This order is deterministic 
# and matches perfectly how S (similarity matrix) was produced previously.
# Concatenating `E_cat` using this exact sequence ensures that the features 
# align with the pre-computed pair-wise relationships.

# Standardization prevents features with larger scales (e.g. dimensions in embeddings 
# or similarities) from dominating the gradients, aiding MLP convergence. Disabling 
# it evaluates if the raw absolute magnitudes hold important semantic meaning.
# It can be toggled by setting `APPLY_STANDARDIZATION` to False.

#%% Helper Functions

def standardize_train_test(train_feat: torch.Tensor, test_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize features using mean and std computed ONLY on the training set
    to prevent data leakage into the test set.
    """
    mean = train_feat.mean(dim=0, keepdim=True)
    std = train_feat.std(dim=0, keepdim=True)
    
    # Add epsilon to prevent division by zero
    train_feat_std = (train_feat - mean) / (std + 1e-6)
    test_feat_std = (test_feat - mean) / (std + 1e-6)
    return train_feat_std, test_feat_std


def get_adapter_model(input_dim: int, num_classes: int = 10) -> nn.Module:
    """
    Creates a simple MLP adapter: Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear
    """
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )

def get_linear_baseline(input_dim: int, num_classes: int = 10) -> nn.Module:
    """
    Creates a single-layer linear baseline: Linear(input_dim, num_classes).
    Serves as a lower-bound reference for the MLP adapters.
    """
    return nn.Sequential(
        nn.Linear(input_dim, num_classes)
    )

def train_one_adapter(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor) -> nn.Module:
    """
    Trains an adapter using the given features and labels on the training half.
    """
    model.to(DEVICE)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
    return model

def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Evaluates the model on the test features and returns test accuracy.
    """
    model.eval()
    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    return correct / total if total > 0 else 0.0

#%% Main Processing Loop

def main():
    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Configuration parsing
    config_path = ensemble_utils.get_ensemble_config_path_from_cli()
    
    # Iterate over ensembles
    for ens in ensemble_utils.iter_ensemble_inference_data_from_config(config_path):
        ensemble_name = ens["ensemble_name"]
        run_names = ens["run_names"]
        M = len(run_names)
        
        print(f"\n" + "="*50)
        print(f"Processing ensemble: {ensemble_name} ({M} models)")
        
        # 2. Load inference embeddings and labels
        # The inference_data is now a dict mapping run_name to the data block
        # We iterate in the order of run_names to ensure deterministic concatenation order
        models_data = [ens["inference_data"][rn] for rn in run_names]
        
        if M == 0 or len(models_data) == 0:
            print("No models found, skipping.")
            continue
            
        labels = models_data[0]["labels"]
        N = labels.shape[0]
        
        emb_list = []
        for i, m_data in enumerate(models_data):
            # Assert N matches across all fields and models
            assert m_data["labels"].shape[0] == N, f"Model {i} has different number of samples"
            assert m_data["embeddings"].shape[0] == N, f"Model {i} has different number of embeddings"
            
            # Assert labels universally match the first model
            assert torch.equal(labels, m_data["labels"]), f"Model {i} labels do not match baseline!"
            
            emb = m_data["embeddings"]
            emb_list.append(emb)
            
        # Ensure uniform embedding dimensions based on the first model
        emb_dim = emb_list[0].shape[1]
        for i, emb in enumerate(emb_list):
            assert emb.shape[1] == emb_dim, f"Model {i} expected embedding dim {emb_dim}, got {emb.shape[1]}"
            
        print(f"Loaded {M} components. N={N}, Embedding dim={emb_dim}.")
        
        # 3. Concatenate embeddings
        E_cat = torch.cat(emb_list, dim=1)  # Shape: (N, emb_dim*M)
        assert E_cat.shape == (N, emb_dim * M)
        
        # 4. Load similarity profiles
        ensemble_dir = ensemble_utils.get_ensemble_path_by_name(ensemble_name, save_dir=env.ENSEMBLES_ROOT)
        sim_path = os.path.join(ensemble_dir, "similarity_profiles.pt")
        
        if not os.path.exists(sim_path):
            print(f"Similarity file not found at {sim_path}, skipping.")
            continue
            
        sim_data = torch.load(sim_path, weights_only=False)

        # Validate run_names ordering matches between similarity profiles and
        # the current config.  A mismatch means the S columns would be misaligned
        # with the concatenated embeddings.
        if "run_names" in sim_data:
            assert sim_data["run_names"] == run_names, (
                f"Run names mismatch!  Similarity profiles were computed with "
                f"{sim_data['run_names']} but current config has {run_names}.  "
                f"Re-run exp_similarityprofiles.py to regenerate."
            )

        # Use ONLY exact required dictionary key
        S = sim_data["rdm_mats_remove"]  # Shape: (N, D_sim)
        
        assert S.shape[0] == N, f"Similarity profiles N={S.shape[0]} doesn't match embeddings N={N}"
        
        D_sim = S.shape[1]
        expected_d_sim = M * (M - 1) // 2
        assert D_sim == expected_d_sim, f"Expected D_sim={expected_d_sim}, got {D_sim}"
        
        # 5. Build feature matrices
        X_embed = E_cat
        X_embed_sim = torch.cat([E_cat, S], dim=1)
        assert X_embed_sim.shape == (N, emb_dim * M + D_sim)
        
        # 6. Split: Exact half split by index
        split_point = N // 2
        train_idx = slice(0, split_point)
        test_idx = slice(split_point, N)
        
        X_embed_train = X_embed[train_idx]
        X_embed_test = X_embed[test_idx]
        
        X_embed_sim_train = X_embed_sim[train_idx]
        X_embed_sim_test = X_embed_sim[test_idx]
        
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        N_train = X_embed_train.shape[0]
        N_test = X_embed_test.shape[0]
        
        # 7. Optionally standardize features (using TRAIN stats to prevent leakage)
        if APPLY_STANDARDIZATION:
            X_embed_train, X_embed_test = standardize_train_test(X_embed_train, X_embed_test)
            X_embed_sim_train, X_embed_sim_test = standardize_train_test(X_embed_sim_train, X_embed_sim_test)
            
        # 8. Train & eval adapters
        input_dim_embed = emb_dim * M
        input_dim_embed_sim = emb_dim * M + D_sim
        
        adapter_embed = get_adapter_model(input_dim_embed)
        adapter_embed_sim = get_adapter_model(input_dim_embed_sim)
        linear_embed = get_linear_baseline(input_dim_embed)
        linear_embed_sim = get_linear_baseline(input_dim_embed_sim)
        
        print("Training Linear Baseline A (Embeddings Only)...")
        linear_embed = train_one_adapter(linear_embed, X_embed_train, y_train)
        
        print("Training Linear Baseline B (Embeddings + Similarity)...")
        linear_embed_sim = train_one_adapter(linear_embed_sim, X_embed_sim_train, y_train)
        
        print("Training Adapter A (Embeddings Only)...")
        adapter_embed = train_one_adapter(adapter_embed, X_embed_train, y_train)
        
        print("Training Adapter B (Embeddings + Similarity)...")
        adapter_embed_sim = train_one_adapter(adapter_embed_sim, X_embed_sim_train, y_train)
        
        acc_linear_embed = evaluate(linear_embed, X_embed_test, y_test)
        acc_linear_embed_sim = evaluate(linear_embed_sim, X_embed_sim_test, y_test)
        acc_embed = evaluate(adapter_embed, X_embed_test, y_test)
        acc_embed_sim = evaluate(adapter_embed_sim, X_embed_sim_test, y_test)
        
        print(f"Summary for {ensemble_name}:")
        print(f"  M Models         : {M}")
        print(f"  S Dimensions     : {D_sim}")
        print(f"  N_train          : {N_train}")
        print(f"  N_test           : {N_test}")
        print(f"  Linear (Embed)   : {acc_linear_embed:.4f}")
        print(f"  Linear (Emb+Sim) : {acc_linear_embed_sim:.4f}")
        print(f"  Acc (Embed Only) : {acc_embed:.4f}")
        print(f"  Acc (Embed+Sim)  : {acc_embed_sim:.4f}")
        
        # 9. Save Weights + Metrics
        adapter_embed_path = os.path.join(ensemble_dir, "adapter_embed.pt")
        adapter_embed_sim_path = os.path.join(ensemble_dir, "adapter_embed_sim.pt")
        linear_embed_path = os.path.join(ensemble_dir, "linear_baseline_embed.pt")
        linear_embed_sim_path = os.path.join(ensemble_dir, "linear_baseline_embed_sim.pt")
        
        torch.save(adapter_embed.state_dict(), adapter_embed_path)
        torch.save(adapter_embed_sim.state_dict(), adapter_embed_sim_path)
        torch.save(linear_embed.state_dict(), linear_embed_path)
        torch.save(linear_embed_sim.state_dict(), linear_embed_sim_path)
        
        metrics = {
            "ensemble_name": ensemble_name,
            "run_names": run_names,
            "M": M,
            "emb_dim": emb_dim,
            "D_sim": D_sim,
            "N": N,
            "split_point": split_point,
            "split_sizes": {"train": N_train, "test": N_test},
            "adapter_architecture": {
                "embed_only": f"{input_dim_embed} -> 128 -> 64 -> 10",
                "embed_sim": f"{input_dim_embed_sim} -> 128 -> 64 -> 10",
            },
            "linear_baseline_architecture": {
                "embed_only": f"{input_dim_embed} -> 10",
                "embed_sim": f"{input_dim_embed_sim} -> 10",
            },
            "training_hyperparams": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "AdamW",
                "grad_clip": 1.0,
                "seed": 42,
            },
            "APPLY_STANDARDIZATION": APPLY_STANDARDIZATION,
            "final_test_accuracies": {
                "acc_linear_embed": acc_linear_embed,
                "acc_linear_embed_sim": acc_linear_embed_sim,
                "acc_embed": acc_embed,
                "acc_embed_sim": acc_embed_sim
            },
            "created": datetime.now(timezone.utc).isoformat(),
        }
        
        metrics_path = os.path.join(ensemble_dir, "adapters_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Saved artifacts to {ensemble_dir}")
        
        # 10. Memory Management
        # Delete active variables and empty cache before next ensemble
        del models_data, emb_list, E_cat, S, X_embed, X_embed_sim
        del X_embed_train, X_embed_test, X_embed_sim_train, X_embed_sim_test
        del y_train, y_test, adapter_embed, adapter_embed_sim, linear_embed, linear_embed_sim
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
