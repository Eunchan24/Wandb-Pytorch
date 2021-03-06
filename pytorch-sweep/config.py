import math
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hyperparameter_defaults  = {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 0.0005,
        'activation': 'relu',
        'optimizer': 'adam',
        'hidden_nodes' : 120,
        'conv1_channels' : 5,
        'conv2_channels' : 16,
        'seed': 42
    }

sweep_config = {
    'name' : 'random-test',
    'method': 'random',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize'   
        },
    'parameters' : {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
            },
        'epochs': {
            'values': [100]
            },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },
        'batch_size': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(32),
            'max': math.log(256),
            }
        },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }


train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
