import numpy as np

filename="E:\for practice game\CNN\Conventional-Neural-Network\dataset\train.npy"
#
with open(filename, 'rb') as f:
   f.read(16)
   data=np.load(filename)
   print(data.shape)# the ins