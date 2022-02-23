from dataset import CIFAR10_Dataset
from model import ConvNet
from optimize import build_optimizer
from utils import train_epoch

import wandb
import config

def train():
    wandb.init(config=config.hyperparameter_defaults)
    w_config = wandb.config
    
    print('==> Preprocess data..')
    train_loader, test_loader = CIFAR10_Dataset(w_config.batch_size, config.train_transform)
    print(w_config)
    print('==> Model load..')
    model = ConvNet(w_config).to(config.DEVICE)
    optimizer = build_optimizer(model, w_config.optimizer, w_config.learning_rate)

    wandb.watch(model, log='all')

    for epoch in range(w_config.epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, wandb)
        print(f"TRAIN: EPOCH {epoch + 1:04d} / {w_config.epochs:04d} | Epoch LOSS {avg_loss:.4f}")
        wandb.log({'Epoch': epoch, "loss": avg_loss, "epoch": epoch})     

sweep_id = wandb.sweep(config.sweep_config, project="sweeps-CIFA10", entity='eckim')
wandb.agent(sweep_id, train, count=20)