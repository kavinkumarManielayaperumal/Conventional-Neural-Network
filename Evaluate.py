import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Model import CNN_Model
import numpy as np
import pickle
from sklearn.metrics import classification_report
 
 
def load_the_model(filename="cnn_model.pth",depth=3,number_of_classes=10):
     model=CNN_Model(depth=depth,number_of_classes=number_of_classes)
    
     print("\n🔹 Expected model layers:")
     print(model.state_dict().keys())
     
     model.load_state_dict(torch.load(filename,weights_only=True))
     model=model.eval()
     return model
 
def evaluate_model(images,labels):
    with torch.no_grad():#no gradient is calculated
        model=load_the_model()# the load the model with trained weights
        output=model(images)# load model is used to predict the output
        _ ,predicates=torch.max(output,1) # here the max value of the output is calculated and the index of the max value is stored in the predicates
        y_ture=labels.numpy()
        y_pred=predicates.numpy()
    return y_ture,y_pred


def accuracy(images,labels):
    y_true, y_pred = evaluate_model(images, labels)
    acc = (y_true == y_pred).sum() / len(y_true) * 100
    print(f"Accuracy: {acc:.2f}%")
    return acc, y_true, y_pred


test_file=r"E:\for practice game\CNN\Conventional-Neural-Network\dataset\cifar-10-python\cifar-10-batches-py"
with open(test_file+"\\test_batch","rb") as f:
    data=pickle.load(f,encoding="bytes")
    images=data[b'data']
    labels=data[b'labels']
    images=images.reshape(-1,3,32,32)
    labels=np.array(labels)
    print(f"Images shape :{images.shape}")
    print(f"Labels shape:{labels.shape}")
    images=images.astype("float32")/255

 # so before passing the images to the model we need to convert the images into tensot 
 
images=torch.tensor(images,dtype=torch.float32)
labels=torch.tensor(labels,dtype=torch.long)

print(f"Images shape :{images.shape}")
print(f"Labels shape: {labels.shape}")

accuracy_of_the_model,y_true,y_pred=accuracy(images,labels)
print(f"Accuracy of the model:{accuracy_of_the_model:.2f}%")

print("classification report")
print(classification_report(y_true,y_pred))

np.savetxt("predictions.txt", y_pred, fmt="%d")
print("Predictions saved to predictions.txt")



