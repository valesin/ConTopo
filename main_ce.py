import argparse
from argparse import BooleanOptionalAction
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset
from utils.train import AverageMeter, save_checkpoint, unwrap, accuracy, tb_logger, grad_norm, split_cifar10_train_val_indices
import torch
import torch.backends.cudnn as cudnn
from networks.shallowCNN import LinearShallowCNN
from networks.modified_ResNet18 import LinearResNet18
from losses.topographic import Global_Topographic_Loss, Local_WS_Loss
import time
import os
import sys
import torch.optim as optim


def parse_arguments():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--trial', type=int, default=0, help='trial number for multiple runs (used in naming folders)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')

    # Topopgaphic Loss settings
    parser.add_argument('topography_type', type=str, choices=['global', 'ws'], help='type of topographic loss to use')
    parser.add_argument('--topographic_loss_rho', type=float, default=0.05, help='balancing factor of the two losses')
    parser.add_argument('--topology', type=str, default='grid', choices=['grid', 'torus'], help='topology of the neuron grid (default: grid)')

    # Optimization settings
    parser.add_argument('--epochs', type=int, default=125, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate (scaled for larger batch)')

    # Model Settings
    parser.add_argument('model_type', type=str, choices=['shallowcnn', 'resnet18'], help='type of model to use')
    parser.add_argument('--embedding_dim', type=int, default=256, help='dimension of the embedding space')
    parser.add_argument('--use_dropout', action=BooleanOptionalAction, default=True, help='use dropout')
    parser.add_argument('--p_dropout', type=float, default=0.5, help='dropout probability (if applicable)')

    arguments = parser.parse_args()

    subdir = 'ResNet18' if arguments.model_type == 'resnet18' else 'ShallowCNN'
    arguments.model_folder = f'./save/{subdir}/models'
    arguments.tensorboard_folder = f'./save/{subdir}/tensorboard/ce'
    arguments.dataset_folder = './dataset'
    arguments.save_freq = max(1, arguments.epochs // 10)  # Save every 10% of epochs, rounded up
    arguments.num_classes = 10

    arguments.model_name = 'crossentropy_{}topo_{}_{}embdims_{}rho_{}epochs_{}bsz_{}nwork_{}lr_{}dropout'.format(
        arguments.topography_type,
        arguments.topology,
        arguments.embedding_dim,
        arguments.topographic_loss_rho, 
        arguments.epochs,
        arguments.batch_size,
        arguments.num_workers,
        arguments.learning_rate,
        arguments.p_dropout if arguments.use_dropout else 0.0,
    )

    run_name = f"trial_{arguments.trial:02d}"

    arguments.tensorboard_folder = os.path.join(arguments.tensorboard_folder, arguments.model_name, run_name)
    os.makedirs(arguments.tensorboard_folder, exist_ok=True)

    arguments.model_folder = os.path.join(arguments.model_folder, arguments.model_name, run_name)
    os.makedirs(arguments.model_folder, exist_ok=True)

    return arguments

def cifar10_loader(arguments):

    # Use standard normalization and data augmentation for CIFAR-10
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Data augmentations for more diversity in training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 45k/5k split from the CIFAR-10 train split (500 per class for val)
    train_indices, val_indices = split_cifar10_train_val_indices(arguments.dataset_folder, val_per_class=500)

    # Datasets with appropriate transforms
    train_dataset = datasets.CIFAR10(root=arguments.dataset_folder, train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(root=arguments.dataset_folder, train=True, transform=val_transform, download=True)
    test_dataset = datasets.CIFAR10(root=arguments.dataset_folder, train=False, download=True, transform=val_transform)

    # Subsets for our split
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def setup_model(arguments):

    # Select the proper model
    if arguments.model_type == 'shallowcnn':
        model = LinearShallowCNN(emb_dim=arguments.embedding_dim, num_classes=arguments.num_classes, ret_emb=True, use_dropout=arguments.use_dropout, p_dropout=arguments.p_dropout)
    elif arguments.model_type == 'resnet18':
        model = LinearResNet18(emb_dim=arguments.embedding_dim, num_classes=arguments.num_classes, ret_emb=True, use_dropout=arguments.use_dropout, p_dropout=arguments.p_dropout)

    # Define the Cross-Entropy loss
    task_loss = torch.nn.CrossEntropyLoss()

    # Select the topographic loss type
    if arguments.topography_type == 'global':
        topographic_loss = Global_Topographic_Loss(weight=1.0, emb_dim=arguments.embedding_dim)
    elif arguments.topography_type == 'ws':
        topographic_loss = Local_WS_Loss(weight=1.0, topology=arguments.topology)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        if isinstance(task_loss, torch.nn.Module):
            task_loss = task_loss.cuda()
        if isinstance(topographic_loss, torch.nn.Module):
            topographic_loss = topographic_loss.cuda()
        cudnn.benchmark = True

    return model, task_loss, topographic_loss

def train(train_loader, model, task_loss, topographic_loss, optimizer, epoch, arguments):
    model.train()  # training mode
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    topographic_losses = AverageMeter()
    task_losses = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    lambda_hat_meter = AverageMeter()

    # Dynamic loss balancing:
    # - 0 < rho < 1  → task loss dominates
    # - rho = 1      → equal weighting
    # - rho > 1      → topographic loss dominates
    # eps avoids div-by-zero; beta is EMA factor for lambda_hat
    rho = arguments.topographic_loss_rho    
    beta = 0.1
    eps = 1e-8
    lambda_max = 1e4
    lambda_hat = None  # smoothed scale for topo loss

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)  # data loading time

        device = next(model.parameters()).device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = labels.shape[0]

        # forward: encoder returns embeddings and logits
        embeddings, features = model(images)

        # task loss on logits
        task_loss_value = task_loss(features, labels)
        
        # topographic term + parameters used for grad-norm measurement
        if arguments.topography_type == 'ws':
            # local WS loss acts on the final linear classifier
            linear_layer = model.encoder.module.fc if isinstance(model, torch.nn.DataParallel) else model.encoder.fc
            topographic_loss_value = topographic_loss(linear_layer=linear_layer)
            measure_params = list(linear_layer.parameters())
        elif arguments.topography_type == 'global':
            # global topo loss acts on embeddings
            topographic_loss_value = topographic_loss(embeddings)
            if isinstance(model, torch.nn.DataParallel):
                measure_params = [p for p in model.encoder.module.parameters() if p.requires_grad]
            else:
                measure_params = [p for p in model.encoder.parameters() if p.requires_grad]
        else:
            measure_params = [p for p in model.parameters() if p.requires_grad]

        # Grad-norm matching: scale topo loss to task loss magnitude
        nt = grad_norm(task_loss_value, measure_params)
        np_ = grad_norm(topographic_loss_value, measure_params)
        target_lambda = (rho * nt / (np_ + eps)).clamp(0.0, lambda_max).detach()
        if lambda_hat is None:
            lambda_hat = target_lambda
        else:
            lambda_hat = (1 - beta) * lambda_hat + beta * target_lambda  # smooth scaling
        # Track lambda_hat value
        lambda_hat_meter.update(float(lambda_hat.detach().cpu()), bsz)

        # combined objective
        loss = task_loss_value + lambda_hat.detach() * topographic_loss_value

        # meters
        losses.update(loss.item(), bsz)
        task_losses.update(task_loss_value.item(), bsz)
        topographic_losses.update(topographic_loss_value.item(), bsz)

        # compute top-1 accuracy (%) on logits
        acc1 = accuracy(features, labels, topk=(1,))[0]
        acc.update(acc1.item() if hasattr(acc1, "item") else float(acc1), bsz)

        # standard optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # timings
        batch_time.update(time.time() - end)
        end = time.time()

        # periodic training log (shows current and running averages + lambda)
        if (idx + 1) % arguments.print_freq == 0:
            lam_val = float(lambda_hat.detach().cpu()) if lambda_hat is not None else 1.0
            print(f'Epoch: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Topographic Loss {topographic_losses.val:.4f} ({topographic_losses.avg:.4f})\t'
                  f'Task Loss {task_losses.val:.4f} ({task_losses.avg:.4f})\t'
                  f'Lambda {lam_val:.4f}\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Acc@1 {acc.val:.3f} ({acc.avg:.3f})')
            sys.stdout.flush()

    return losses.avg, topographic_losses.avg, task_losses.avg, lambda_hat_meter.avg, acc.avg

def validate(val_loader, model, criterion, arguments):
    model.eval()  # eval mode

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():  # no grads during evaluation
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            device = next(model.parameters()).device
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz = labels.shape[0]

            # forward + CE loss on logits
            output = model(images)
            loss = criterion(output, labels)

            # update loss and accuracy
            losses.update(loss.item(), bsz)
            _, predicted = output.max(1)
            correct = predicted.eq(labels).sum().item()
            acc_value = correct / bsz
            acc.update(acc_value, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # periodic validation log
            if idx % arguments.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, acc=acc))

    print(' * Acc@1 {acc.avg:.3f}'.format(acc=acc))
    return losses.avg, acc.avg

def main():
    arguments = parse_arguments()

    train_loader, val_loader, test_loader = cifar10_loader(arguments)

    model, task_loss, topographic_loss = setup_model(arguments)

    optimizer = optim.Adam(model.parameters(), lr=arguments.learning_rate)

    logger = tb_logger.Logger(logdir=arguments.tensorboard_folder, flush_secs=2)

    # Track best model by validation accuracy
    best_val_acc = 0.0
    last_val_acc = 0.0
    epochs_no_improve = 0
    es_patience = 25  # early stopping patience based on validation accuracy

    # Adapter that makes the model return logits only (validate() expects logits)
    class LogitsOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                return out[1]
            return out

    logits_model = LogitsOnly(model)

    for epoch in range(1, arguments.epochs + 1):
        prev_best = best_val_acc

        time1 = time.time()
        # Train end-to-end with CE; topo term handled inside train()
        avg_loss, avg_topoloss, avg_taskloss, avg_lambda_hat, avg_train_acc = train(
            train_loader, model, task_loss, topographic_loss, optimizer, epoch, arguments
        )
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Log training metrics
        logger.log_value('train_total_loss', avg_loss, epoch)
        logger.log_value('train_task_loss', avg_taskloss, epoch)
        logger.log_value('train_topographic_loss', avg_topoloss, epoch)
        logger.log_value('lambda_hat', avg_lambda_hat, epoch)
        logger.log_value('train_acc', avg_train_acc, epoch)

        # Validation on logits
        val_loss, val_acc = validate(val_loader, logits_model, task_loss, arguments)
        last_val_acc = val_acc
        logger.log_value('val_loss', val_loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        # Save periodic checkpoint (model + optimizer + metrics)
        if (epoch % arguments.save_freq) == 0:
            state = {
                'stage': 'e2e',
                'epoch': epoch,
                'state_dict': unwrap(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(arguments),
                'metrics': {
                    'train_total_loss': avg_loss,
                    'train_task_loss': avg_taskloss,
                    'train_topographic_loss': avg_topoloss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }
            }
            ckpt_path = os.path.join(arguments.model_folder, f'e2e_epoch{epoch:04d}.pth')
            save_checkpoint(ckpt_path, state)

        # Save best checkpoint by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'stage': 'e2e',
                'epoch': epoch,
                'state_dict': unwrap(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(arguments),
                'metrics': {
                    'train_total_loss': avg_loss,
                    'train_task_loss': avg_taskloss,
                    'train_topographic_loss': avg_topoloss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }
            }
            best_path = os.path.join(arguments.model_folder, 'e2e_best.pth')
            save_checkpoint(best_path, best_state)

        # Early stopping check based on validation accuracy
        if best_val_acc > prev_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(f'[CE] Early stopping at epoch {epoch} (no val acc improvement for {es_patience} epochs).')
                break

    # Always save a final "last" snapshot for e2e training (for resume/export)
    final_e2e = {
        'stage': 'e2e',
        'epoch': arguments.epochs,
        'state_dict': unwrap(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(arguments),
        'val_acc': last_val_acc,
    }
    save_checkpoint(os.path.join(arguments.model_folder, 'e2e_last.pth'), final_e2e)

    # Final single evaluation on the held-out test set using the best-val checkpoint
    best_path = os.path.join(arguments.model_folder, 'e2e_best.pth')
    try:
        device = next(model.parameters()).device
        ckpt = torch.load(best_path, map_location=device)
        unwrap(model).load_state_dict(ckpt['state_dict'])
    except Exception as e:
        print(f'Warning: failed to load best checkpoint from {best_path}: {e}')

    test_loss, test_acc = validate(test_loader, logits_model, task_loss, arguments)
    logger.log_value('test_acc', test_acc, arguments.epochs)
    print('Final Test Acc@1 {:.3f}'.format(test_acc))

    if hasattr(logger, "close"):
        logger.close()

if __name__ == '__main__':
    main()
