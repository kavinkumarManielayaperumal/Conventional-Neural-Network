import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



def to_view_images(train_file,):
   train_data=np.load(train_file,allow_pickle=True)# raw data like images and size , number of images
   
   print(f"Shape of the dataset: {len(train_data)}") 
   print(f"type of the images:{type(train_data)}")
   
   return train_data


def to_view_labels(labels_file):
   labels_pd=pd.read_csv(labels_file)
   print(f"shape of the labels:{labels_pd.shape}")
   print(f"type of the labels:{type(labels_pd)}")
   if labels_pd.shape[1]==2:
      print(f"checking the column names:{labels_pd.head()}")
      print(f"checking the column names:{labels_pd.columns}")
      labels=labels_pd.iloc[:,1]
   else:
      labels=labels_pd
   return labels
   
   
   
if __name__=="__main__":
   train_file=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\train.npy"
   labels_file=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\trainLabels.csv"
   image=to_view_images(train_file)
   #labels=to_view_labels(labels_file)
   index=5
   plt.imshow(image[5])
   #plt.title(f"Label:{labels.iloc[index,1]}")# inedx is the row with means th image number and 1 is the column number
   plt.axis("off")
   plt.show()
