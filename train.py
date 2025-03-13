import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
from Model import CNN_Model
from dataset_viewer import data_viewer
from data_loader import data_loader
import matplotlib.pyplot as plt
import numpy as np

def Train_Model(model,data_loader,number_epochs=10, learning_rate=0.001):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.paremeter(),lr=learning_rate)
    training_loss=[]
    
    for epoch in range(number_epochs):
        total_loss=0
    