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
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
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
        avg_loss=total_loss/len(data_loader)
        training_loss.append(avg_loss)
        print(f'Epoch:{epoch}, Batch:{batch_size}, Avg Loss:{avg_loss}')
    plt.plot(range(1,number_epochs+1),training_loss,marker='o',linestyle='--',color='r')
    plt.xlabel('Number of Epochs')
    plt.ylabel("loss")
    plt.title("Training_LOss")
    plt.grid(True)
    plt.show()
    save_model(model,file_name='cnn_model.pth')
        


def save_model(model,file_name='cnn_model.pth'):
    torch.save(model.state_dict(),file_name)
    print(f"Model is saved as {file_name}")
    
if __name__=='__main__':
    train_folder=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\cifar-10-python\cifar-10-batches-py"
    images,labels=data_viewer(train_folder)
    loaded_images=data_loader(images,labels,batch_size=32)
    
    model=CNN_Model(depth=3,number_of_classes=10)
    
    Train_Model(model,loaded_images,number_epochs=10,learning_rate=0.001)
    
    print("Training is done")