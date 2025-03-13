from importlib.abc import Loader
import torch
from dataset_viewer import data_viewer
from torch.utils.data import DataLoader, Dataset

class cifarDataset(Dataset):
    def __init__(self,images,labels):
        self.images=torch.tensor(images,dtype=torch.float32)# images is already in the form 0f 4D array, in the float32 but for confirmation onces again we covert int to float in the form of tensor
        self.labels=torch.tensor(labels,dtype=torch.long)# here the data type is long because the labels are in the form of integer
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        images=self.images[idx]
        labels=self.labels[idx]
        return images, labels
    
def data_loader(images,labels,batch_size):
    dataset=cifarDataset(images,labels)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return loader

if __name__=="__main__":
    data_folder=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\cifar-10-python\cifar-10-batches-py"
    images,labels=data_viewer(data_folder)
    
    loaded_images=data_loader(images,labels,batch_size=32)
    
    for images,labels in loaded_images:
        print(images.shape)
        print(labels.shape)
        break# we dont want to print all batchs of images and labels, so we break the loop after first batch of images and labels
