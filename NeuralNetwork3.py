#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:18:23 2022

@author: ravichandrabasireddy
"""

import numpy as np
import pandas as pd
import sys

print(sys.argv[1],sys.argv[2],sys.argv[3])

columns=['Pixel'+str(x) for x in range(784)]

train_images=pd.read_csv(sys.argv[1],header=None)
train_images.columns=columns
train_labels=pd.read_csv(sys.argv[2],header=None)
train_labels.columns=['Label']
test_images=pd.read_csv(sys.argv[3],header=None)
test_images.columns=columns
test_labels=pd.read_csv('./test_label.csv',header=None)
test_labels.columns=['Label']

train_data=train_labels.join(train_images)

train_data=np.array(train_data)
np.random.shuffle(train_data)
n_rows,n_columns=train_data.shape

train_data=train_data.T
y_train=train_data[0]
x_train=train_data[1:10000]
x_train=x_train/255.
x_train_rows,x_train_columns=x_train.shape

test_data=np.array(test_images)
test_data=test_data.T
x_test=test_data/255.
y_test=np.array(test_labels)
y_test=y_test.T


def initialize_neural_network():
    W1=np.random.rand(32,784)-0.5
    b1=np.random.rand(32,1)-0.5
    W2=np.random.rand(16,32)-0.5
    b2=np.random.rand(16,1)-0.5
    W3=np.random.rand(10,16)-0.5
    b3=np.random.rand(10,1)-0.5
    return W1,b1,W2,b2,W3,b3

def sigmoid_activation_function(X):
    return 1/(1+np.exp(-X))

def softmax_activation_function(X):
    return np.exp(X)/sum(np.exp(X))

def forward_propagation(W1,b1,W2,b2,W3,b3,X):
    Z1=np.dot(W1,X)+b1
    A1=sigmoid_activation_function(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid_activation_function(Z2)
    Z3=np.dot(W3,A2)+b3
    A3=softmax_activation_function(Z3)
    return Z1,A1,Z2,A2,Z3,A3

def one_hot_encoding(y_pred):
    one_hot_encoded_Y = np.zeros((y_pred.size, y_pred.max() + 1))
    one_hot_encoded_Y[np.arange(y_pred.size), y_pred] = 1
    one_hot_encoded_Y = one_hot_encoded_Y.T
    return one_hot_encoded_Y

def back_propogation(Z1,A1,W1,Z2,A2,W2,Z3,A3,W3,X,Y):
    one_hot_encoded_Y = one_hot_encoding(Y)
    dZ3=A3-one_hot_encoded_Y
    dW3=(1./x_train_rows)*np.dot(dZ3,A2.T)
    db3=(1./x_train_rows)*np.sum(dZ3)
    dA2=np.dot(W3.T,dZ3)
    dZ2=dA2*sigmoid_activation_function(Z2)*(1-sigmoid_activation_function(Z2))
    dW2=(1./x_train_rows)*np.dot(dZ2,A1.T)
    db2=(1./x_train_rows)*np.sum(dZ2)
    dA1=np.dot(W2.T,dZ2)
    dZ1=dA1*sigmoid_activation_function(Z1)*(1-sigmoid_activation_function(Z1))
    dW1=(1./x_train_rows)*np.dot(dZ1,X.T)
    db1=(1./x_train_rows)*np.sum(dZ1)
    return dW1,db1,dW2,db2,dW3,db3

def update_parameters(W1,b1,dW1,db1,W2,b2,dW2,db2,W3,b3,dW3,db3,alpha):
    W1=W1-alpha*dW1
    b1=b1-alpha*db1
    W2=W2-alpha*dW2
    b2=b2-alpha*db2
    W3=W3-alpha*dW3
    b3=b3-alpha*db3
    return W1,b1,W2,b2,W3,b3

def get_predictions(A3):
    return np.argmax(A3,0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_decent(X,Y,alpha,iterations):
    W1,b1,W2,b2,W3,b3=initialize_neural_network()
    for i in range(iterations):
        Z1,A1,Z2,A2,Z3,A3=forward_propagation(W1,b1,W2,b2,W3,b3,X)
        dW1,db1,dW2,db2,dW3,db3=back_propogation(Z1,A1,W1,Z2,A2,W2,Z3,A3,W3,X,Y)
        W1,b1,W2,b2,W3,b3=update_parameters(W1,b1,dW1,db1,W2,b2,dW2,db2,W3,b3,dW3,db3,alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1,b1,W2,b2,W3,b3


W1,b1,W2,b2,W3,b3=gradient_decent(x_train,y_train,0.10,500)
Z1,A1,Z2,A2,Z3,A3=forward_propagation(W1,b1,W2,b2,W3,b3,x_test)
predictions = get_predictions(A3)
accuracy=get_accuracy(predictions, y_test)
print("accuracy",accuracy)