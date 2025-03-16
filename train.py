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
        for batch_size,(images,labels) in enumerate(data_loader):
            optimizer.zero_grad()# this pytorch functdion for every batch the gradient is zero
            output=model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()# this pytorch function id used to upadate the weights
            total_loss+=loss.item()# item turn the loss into the scalar value beacuse the loss are in the form of tensor
            