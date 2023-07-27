from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import Utils
from torchsummary import summary

def print_model_summary(model, device):

    device = Utils.get_device()
    cifar_model = model.to(device)
    return summary(cifar_model, input_size=(3, 32, 32))

def experiment(model,train_loader, test_loader,optimizer,schedular,criterion,EPOCHS):

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    lrs = []

    device = Utils.get_device()
    model = model.to(device)

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        Utils.train(model, device, train_loader, optimizer, epoch,train_losses,train_acc,schedular,criterion,lrs)
        Utils.test(model, device, test_loader,test_losses,test_acc,criterion)   

    return (model, train_acc, train_losses, test_acc, test_losses)
