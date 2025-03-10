import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

#pickle file are not lilke model file from machine learning model , this file is used to store the array of data , raw scalar data , raw image data with key value pair(in this case key is name of the image like label1, label2, label3 and value is the image data)
#pickle file is used to store the data in the from of dictionary


def load_the_pickle_file(filename):
   with open(filename,"rb") as f:# same as usual file handling with rb mode to read the binary file format
      data=pickle.load(f)
      images=data[b'data']# pixel values in the form of 1D array and b is used to convert the string to bytes and , here the shape will be [numbers of images, 3072]
      labels=data[b'labels']# labels of the images in the form of list in integer format
      
      images=images.reshape(-1,3,32,32)# reshaping the images in the form of 3D array , here the shape will be [ number of images, 3,32,32]
      labels=np.array(labels)# this is used to convert the list into numpy array , because we can't use the list in the machine learning model
      
      images=images.transpose(0,2,3,1)# most of the machine learning model accept the image in the form of 4D array and the shape will be [number of images, 32, 32 , 3] , here 4D means each images is 3D array and batch of images is 4D array , you tell combine all the images in the form of 4D array
      
      return images, labels
   

def load_the_batch_files(folder_path):
   train_images=[]# here 5 batch file will be loaded and stored in  the list, shape will be [5, number of images, 32, 32, 3]
   train_labels=[]
   for i in range(1,6):
      filename=folder_path+f"/data_batch_{i}"
      images, labels=load_the_pickle_file(filename)# ecah image in batch file , go to the function and load the images and labels
      train_images.append(images)# every images is stored in the list from the batch files
      train_labels.append(labels)# every labels is stored in the list from the batch files
   train_images=np.concatenate(train_images)# its like all the images are stored in the list and now we are combining all the images in the form of 4d array
   train_labels=np.concatenate(train_labels)
   return train_images, train_labels #


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
