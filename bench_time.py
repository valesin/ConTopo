import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Subset

from utils.train import grad_norm, split_cifar10_train_val_indices
from networks.modified_ResNet18 import LinearResNet18
from losses.topographic import Global_Topographic_Loss

def build_loaders(dataset_folder='./dataset', batch_size=512, num_workers=4):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 45k/5k split from the CIFAR-10 train split (500 per class for val)
    train_indices, _ = split_cifar10_train_val_indices(dataset_folder, val_per_class=500)
    
    train_dataset = datasets.CIFAR10(root=dataset_folder, train=True, transform=train_transform, download=True)
    train_subset = Subset(train_dataset, train_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader

def main():
    # Hardcoded hyperparams matching main_ce.py default args
    epochs = 125
    batch_size = 512
    learning_rate = 0.002
    embedding_dim = 256
    num_classes = 10
    rho = 0.05  # fixed rho (not 0)
    
    print(f"Starting training benchmark with:")
    print(f"  Model: ResNet18 -> CE Loss + Global Topographic Loss")
    print(f"  Rho: {rho}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
    
    train_loader = build_loaders(batch_size=batch_size)
    
    # Setup Model
    model = LinearResNet18(emb_dim=embedding_dim, num_classes=num_classes, ret_emb=True, use_dropout=True, p_dropout=0.5)
    task_loss = torch.nn.CrossEntropyLoss()
    topographic_loss = Global_Topographic_Loss(weight=1.0, emb_dim=embedding_dim)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        task_loss = task_loss.cuda()
        topographic_loss = topographic_loss.cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dynamic loss balancing parameters
    beta = 0.1
    eps = 1e-8
    lambda_max = 1e4
    lambda_hat = None

    print("\nStarting training loop...")
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        epoch_start_time = time.time()
        for idx, (images, labels) in enumerate(train_loader):
            device = next(model.parameters()).device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            embeddings, features = model(images)
            task_loss_value = task_loss(features, labels)
            
            # Global topo loss acts on embeddings
            topographic_loss_value = topographic_loss(embeddings)
            if isinstance(model, torch.nn.DataParallel):
                measure_params = [p for p in model.encoder.module.parameters() if p.requires_grad]
            else:
                measure_params = [p for p in model.encoder.parameters() if p.requires_grad]
                
            # Grad-norm matching: scale topo loss to task loss magnitude
            nt = grad_norm(task_loss_value, measure_params)
            np_ = grad_norm(topographic_loss_value, measure_params)
            target_lambda = (rho * nt / (np_ + eps)).clamp(0.0, lambda_max).detach()
            
            if lambda_hat is None:
                lambda_hat = target_lambda
            else:
                lambda_hat = (1 - beta) * lambda_hat + beta * target_lambda
                
            loss = task_loss_value + lambda_hat.detach() * topographic_loss_value
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:03d}/{epochs} completed in {epoch_time:.2f} seconds.")

    total_time = time.time() - total_start_time
    
    # Format total time into hours, minutes, seconds
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n--- Training Benchmark Completed ---")
    print(f"Total time for {epochs} epochs: {int(hours)}h {int(minutes)}m {seconds:.2f}s ({total_time:.2f} seconds total)")
    print(f"Average time per epoch: {total_time / epochs:.2f} seconds")

if __name__ == '__main__':
    main()
