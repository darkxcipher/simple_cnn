

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
#we convert the MNIST image giles into tensor of 4D 
#(no of images, height, width, color channels )
#change image to tensor
transform = transforms.ToTensor()

#TrainData

train_data = datasets.MNIST(root='./cnn_data', train = True, download=True, transform=transform)

#TestData
test_data = datasets.MNIST(root='./cnn_data', train = False, download=True, transform=transform)

#print(train_data)
#print(test_data)

#create a small batch size for images, lets say 8

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(train_data, batch_size=8, shuffle=False)

#define our CNN Model

"""
--------- 2 layer example---
#decription of convolutional layer
#here its 2 layes

conv1 = nn.Conv2d(1, 6, 3, 1)
conv2 = nn.Conv2d(6, 12, 3, 1)

#grab 1 MNIST image

for i, (X_Train, y_train) in enumerate(train_data):
    break

print(X_Train.shape)

x = X_Train.view(1,1,28,28)
#print(x)


#perform first convultion here Relu
#Relu of our activation function
x = F.relu(conv1(x))
print(x.shape)
#output is torch.Size([1, 6, 26, 26])
#where 1 single image
# 6 is the filter
#output is 26x26, we had defined 28x28 in the convlayer
#the outer pixels are dropped as they dont contain much info in our mnist img

#pass through the pooling layer
#input of x and kernel size 2, pooling size 2
x = F.max_pool2d(x,2,2)
print(x.shape)

#output torch.Size([1, 6, 13, 13])
#here the pooling layer reduces it to 1,6,13,13 from 1,6,26,26
#it has reduced the info as 26/2 = 13

#second conv layer 
x = F.relu(conv2(x))
print(x.shape)
#output torch.Size([1, 12, 11, 11])
#we lose padding again from 13x13 to 11x11

#another pooling layer
x = F.max_pool2d(x,2,2)
print(x.shape)

#output
#torch.Size([1, 12, 5, 5])

----------end----
"""

#Model class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        #Fully connected layer
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,60)
        self.fc3 = nn.Linear(60,10)
    
    def forward(self,X):
        X = F.relu(self.conv1(X))
        # 2x2 kernel, 2  stride
        X = F.max_pool2d(X,2,2)

        X = F.relu(self.conv2(X))
        # 2x2 kernel, 2  stride
        X = F.max_pool2d(X,2,2)

        #here -1 because we can vary batch size
        X = X.view(-1, 16*5*5)

        #fully connected laters
        
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim =1)

#Create an instance of our Model

torch.manual_seed(256)
model = ConvolutionalNetwork().to(device)
print(model)

#loss function and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

import time
start_time = time.time()

#create variables to track things 
epochs = 4
train_losses = []
test_losses = []
train_correct = []
test_correct = []

train_acc = []   # <--- add this
test_acc = []    # <--- and this

#For loops of epocs 2
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    #Train
    for b,(X_train, y_train) in enumerate(train_loader):
        X_train, y_train = X_train.to(device), y_train.to(device) 
        b+=1 # strat our batches at 1
        y_pred = model(X_train)# get predicted values from the training set, not flattened.
        loss = criterion(y_pred, y_train) # compare the predictions to correct answers in y_train
        
        predicted = torch.max(y_pred.data, 1)[1] # add up the number of correct predictions, indexed off the first poinr
        batch_corr = (predicted == y_train).sum() # how many we got correct from this batch 
        trn_corr += batch_corr # t keep track as we go along in the training 

        
        #update our parameters
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print out results
        

        if b%600 == 0:
            print(f'Epoc: {i} Batch: {b} Loss: {loss.item()}')        
    train_losses.append(loss.item())
    train_correct.append(trn_corr)
    train_acc.append(trn_corr / len(train_data))
    #print(next(model.parameters()).device)
    torch.save(model.state_dict(), "mnist_cnn.pth")
    torch.save(model, "mnist_cnn_full.pth")
    print("Model saved to mnist_cnn.pth")

    #Test
    with torch.no_grad(): # NO gradient as we dont want to update it with test data
        for b,(X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(device), y_test.to(device) 
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1] # adding up correct predictions
            tst_corr += (predicted == y_test).sum() # True=1 False=0 sum it up
            
        loss = criterion(y_val, y_test)
            
    test_losses.append(loss.item())
    test_correct.append(tst_corr)
    test_acc.append(tst_corr / len(test_data)) 
            #print(next(model.parameters()).device)


current_time = time.time()
total = current_time - start_time
print(f'Training took: {total/60} minutes')

#lets graph the loss at epochs
# convert tensors to Python floats
'''
train_losses = [tl.item() for tl in train_losses]
test_losses = [tl.item() for tl in test_losses]

plt.plot(range(1, len(train_losses)+1), train_losses, label="Training loss")
plt.plot(range(1, len(test_losses)+1), test_losses, label="Validation loss")

plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

'''
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Validation loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

'''
#grab an image 
test_data[4143][0].reshape(28,28) #tensor image, last digit shows the label 0-9

#pass the image through 
model.eval()
with torch.no_grad():
    new_prediction = model(test_data[4143][0].view(1,1,28,28))

new_prediction.argmarx()

'''