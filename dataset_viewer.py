import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
import torch


#pickle file are not lilke model file from machine learning model , this file is used to store the array of data , raw scalar data , raw image data with key value pair(in this case key is name of the image like label1, label2, label3 and value is the image data)
#pickle file is used to store the data in the from of dictionary


def load_the_pickle_file(filename):
   with open(filename,"rb") as f:# same as usual file handling with rb mode to read the binary file format
      data=pickle.load(f,encoding='bytes')# here the data is loaded in the form of dictionary and encoding is used to convert the string to bytes
      images=data[b'data']# pixel values in the form of 1D array and b is used to convert the string to bytes and , here the shape will be [numbers of images, 3072]
      labels=data[b'labels']# labels of the images in the form of list in integer format
      print(f"Images shape before reshaping or raw shape:{images.shape}")# here the shape will be [number of images, 3072] here 3072 is per image pixel values
      print(f"Labels shape before reshaping or raw shape:{len(labels)}")
      images=images.reshape(-1,3,32,32)# reshaping the images in the form of 3D array , here the shape will be [ number of images, 3,32,32]
      labels=np.array(labels)# this is used to convert the list into numpy array , because we can't use the list in the machine learning model
      
      #images=images.transpose(0,2,3,1)# for the mathplotlib the image should be in the form of 4d array and the shape will be [number of images, 32, 32 , 3] , here 4D means each images is 3D array and batch of images is 4D array , you tell combine all the images in the form of 4D array 
      # images=images.transpose(0,2,3,1).astype(np.uint8)here astype is used to convert ths data type of the image from float to integer if you want to
      
      images=images.astype("float32")/255# here the pixel values are converted into float and divided by 255 to normalzie the pixel values between 0 to 1
      # this called min max scaling normalization
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
   return train_images, train_labels 

def data_viewer(folder_path):
   images,labels=load_the_batch_files(folder_path)
   print(f"Images shape after reshaped:{images.shape}")
   print(f"Labels shape after reshaped:{labels.shape}")
   return images, labels


if __name__=="__main__":
   train_folder=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\cifar-10-python\cifar-10-batches-py"
   
   image,labels=data_viewer(train_folder)
   
   
   
   images=image.transpose(0,2,3,1)#  this transpose for just mathplotlib to show the images
   
   plt.figure(figsize=(5,5))
   plt.imshow(images[10], interpolation="bilinear")
   plt.title(f"Label:{labels[10]}")
   plt.axis("off")
   plt.show()
   

