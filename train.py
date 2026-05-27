"""Training script for the multi-modal remote sensing classification model."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import data
import pandas as pd
import torch
import torch.nn as nn
import argparse
import random
import numpy as np
from utils import calculate_metrics
from test import predicted
import time
from model import Model
from torch.utils.data import ConcatDataset, DataLoader


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Update the optimizer learning rate using a cosine decay schedule."""

    cur_lr = init_lr * 0.5 * (1. + np.cos(np.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def train(train_loader, val_loader, model, criterion, optimizer, epoch, args):
    """Train the model for one epoch and evaluate it on the validation set."""

    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for i, (spectral_data, spatial_data, sar_data, phe_data, labels) in enumerate(train_loader):
        spectral_data, spatial_data, sar_data, phe_data, labels = spectral_data.cuda(), spatial_data.cuda(), sar_data.cuda(), phe_data.cuda(), labels.cuda()

        outputs = model(spectral_data, spatial_data, sar_data, phe_data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * spectral_data.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += spectral_data.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    model.eval()
    val_loss = 0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():

        for spectral_data, spatial_data, sar_data, phe_data, labels in val_loader:
            spectral_data, spatial_data, sar_data, phe_data, labels = spectral_data.cuda(), spatial_data.cuda(), sar_data.cuda(), phe_data.cuda(), labels.cuda()

            outputs = model(spectral_data, spatial_data, sar_data, phe_data)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * spectral_data.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_samples += spectral_data.size(0)

    avg_val_loss = val_loss / val_samples
    avg_val_accuracy = val_correct / val_samples

    print(f'Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')

    return avg_val_loss, avg_val_accuracy


def setup_seed(seed):
    """Set random seeds to improve the reproducibility of model training."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    acc_lis = []
    time_s = time.time()
    for seed in range(0, 10):
        p = argparse.ArgumentParser()
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                       help='initial (base) learning rate', dest='lr')
        p.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
        p.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                       help='weight decay (default: 1e-4)', dest='weight_decay')
        p.add_argument("--input_dim", type=int, default=12)
        p.add_argument("--spatial_dim", type=int, default=3)
        p.add_argument("--sar_dim", type=int, default=2)
        p.add_argument("--phe_dim", type=int, default=5)
        p.add_argument("--hidden_dim", type=int, default=256)
        p.add_argument("--num_layers", type=int, default=2)
        p.add_argument("--output_dim", type=int, default=3)
        p.add_argument("--num_heads", type=int, default=8)
        p.add_argument("--dropout", type=float, default=0.5)
        p.add_argument('--Training_data', type=int, default=70, help='the number of training data.')
        p.add_argument('--Val_data', type=int, default=20, help='the number of validation data.')
        args = p.parse_args()

        setup_seed(seed)
        model_info = {
            "Epochs": args.epochs,
            "Learning Rate": args.lr,
            "Batch Size": args.batch_size,
            "Layers": args.num_layers,
            "Hidden Dimension": args.hidden_dim,
            "Dropout": args.dropout,
            "Traninig Data": args.Training_data,
            "Validation Data": args.Val_data,
            "Number Heads": args.num_heads
        }

        model = Model(spectral_dim=args.input_dim,
                      spatial_dim=args.spatial_dim,
                      sar_dim=args.sar_dim,
                      phenology_dim=args.phe_dim, hidden_dims=args.hidden_dim, num_layers=args.num_layers,
                      num_classes=args.output_dim).cuda()

        TSdata_load = data.TSLoader('D:/1van/ModelC/GuanzhaiSea/CSV/TrainModelC.csv', args.Training_data / 100,
                                         (args.Training_data + args.Val_data) / 100, seed)
        train_dataset, train_augmented_dataset, val_dataset, test_dataset = TSdata_load.return_data()

        combined_train_dataset = ConcatDataset([train_dataset, train_augmented_dataset])

        train_loader = DataLoader(
            combined_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss().cuda()

        best_val_accuracy = 0.0
        global_best_val_accuracy = 0.0
        global_best_model = None

        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, args.lr, epoch, args)

            val_loss, val_accuracy = train(train_loader, val_loader, model, criterion, optimizer, epoch, args)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'D:/1van/ModelC/GuanzhaiSea/Modelsave/1.pth')

                if val_accuracy > global_best_val_accuracy:
                    global_best_val_accuracy = val_accuracy
                    global_best_model = model.state_dict()

            print('current best validation accuracy', best_val_accuracy)

        best_model = Model(spectral_dim=args.input_dim,
                           spatial_dim=args.spatial_dim,
                           sar_dim=args.sar_dim,
                           hidden_dims=args.hidden_dim, num_layers=args.num_layers, num_classes=args.output_dim,).cuda()


        best_model.load_state_dict(torch.load('D:/1van/ModelC/GuanzhaiSea/Modelsave/1.pth', weights_only=True))

        y_true, y_pred = predicted(test_loader, best_model)
        pre_acc = calculate_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy())
        print(pre_acc)
        acc_lis.append(pre_acc)

    torch.save(global_best_model, 'E:/1van/ModelC/GuanzhaiSea/Modelsave/1.pth')

    time_e = time.time()
    print(time_e - time_s)
    acc_lis = np.array(acc_lis)

    mean = np.expand_dims(np.mean(acc_lis, axis=0), axis=0)
    std = np.expand_dims(np.std(acc_lis, axis=0), axis=0)
    acc_lis = np.concatenate((acc_lis, mean, std), axis=0)

    def format_model_info(model_info):
        """Format model hyperparameters for readable table output."""

        formatted_info = {}
        for key, value in model_info.items():
            if isinstance(value, float):
                formatted_info[key] = f"{value:.6f}"
            else:
                formatted_info[key] = str(value)
        return formatted_info


    def print_metrics_table_with_model_info(acc_lis, model_info):
        """Print model settings and accuracy metrics for all runs, including mean and standard deviation."""

        metrics_names = ["OA", "Kappa"]
        num_classes = len(acc_lis[0]) - 10
        for i in range(num_classes):
            metrics_names.append(f"UA Class {i}")
        for i in range(num_classes):
            metrics_names.append(f"Recall Class {i}")
        for i in range(num_classes):
            metrics_names.append(f"F1 Class {i}")

        metrics_names.append("Macro Recall")
        metrics_names.append("Macro F1")

        results = {}

        for i in range(len(acc_lis)):
            if i < len(acc_lis) - 2:
                run_name = f"Run {i + 1}"
            elif i == len(acc_lis) - 2:
                run_name = "Mean"
            else:
                run_name = "Std Dev"

            results[run_name] = [f"{value:.2f}" for value in acc_lis[i]]

        df_metrics = pd.DataFrame(results, index=metrics_names)

        formatted_model_info = format_model_info(model_info)

        df_model_info = pd.DataFrame(formatted_model_info, index=["Model Info"]).T

        print(df_model_info)

        print("\nMetrics Results Table:")
        print(df_metrics)


    print_metrics_table_with_model_info(acc_lis, model_info)