import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def dataset_viewer(train_file,labels_file):
   train_data=np.load(train_file)# raw data like images and size , number of images
   label_data=pd.read_csv(labels_file)#this just labels for the each images
   print(f"shape of the images:{train_data.shape}")
   print(label_data.shape)
   return train_data, label_data
   
   
if __name__=="__main__":
   train_file=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\train.npy"
   labels_file=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\trainLabels.csv"
   images,labels=dataset_viewer(train_file,labels_file)
   index=1
   plt.imshow(images[1])
   #plt.title(f"Label:{labels.iloc[index,0]}")
   plt.axis("off")
   plt.show()
   
   
 