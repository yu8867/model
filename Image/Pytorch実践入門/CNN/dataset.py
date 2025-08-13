import torch
import torch.nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def makedataset(batch_size):
    data_path = "~/Github/model/Image/Pytorch実践入門/datasets"
    cifar10 = datasets.CIFAR10(data_path, train=True, download=True, 
                               transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4915, 0.4813, 0.4468), (0.2470, 0.2435, 0.2616))
                                ])
                                )
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, 
                                   transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4915, 0.4813, 0.4468), (0.2470, 0.2435, 0.2616))
                                        ])
                                    )
                                            

    train_dataloader = DataLoader(cifar10, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=True)
    return train_dataloader, validation_dataloader

if __name__ == '__main__':
    makedataset(32)