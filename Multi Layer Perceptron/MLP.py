import numpy as np
from math import sqrt
import pickle
import pandas as pd
import os

def RELU(x):
    # alternative return x * (x > 0)
    return (np.maximum(0,x))

def RELU_DER(x):
   return np.where(x > 0, 1, 0)

def forward_propagation(A0,W1,B1,W2,B2):
    Z1 = np.dot(W1,A0) + B1
    # Z1 = W1 A0 + B1
    A1 = RELU(Z1)
    Z2 = np.dot(W2,A1) + B2
    # Z2 = W1 A1 + B2
    A2 = RELU(Z2)
    return A1,A2,Z1,Z2


def back_propagation(A0,A1,A2,Y,W2,Z1,Z2):
    dZ2 = 2*(A2 - Y)*RELU_DER(Z2)
    dW2 = np.dot(dZ2,A1.transpose())
    dB2 = dZ2
    dZ1 = np.dot(W2.transpose(),dZ2)*RELU_DER(Z1)
    dW1 = np.dot(dZ1,A0.transpose())
    dB1 = dZ1
    return dW1,dB1,dW2,dB2


def update(dW1,dW2,dB1,dB2,W1,B1,W2,B2,a):
    W1_up = W1 - a*dW1
    B1_up = B1 - a*dB1
    W2_up = W2 - a*dW2
    B2_up = B2 - a*dB2
    return W1_up,B1_up,W2_up,B2_up

def squared_error(A2,Y):
    dY = (A2-Y)
    return pow(dY,2)



def TRAIN_ANN(X,Y,n1,n2,n3):
    alpha = 0.001
    iterations = 30
    Yavg = np.mean(Y)
    def initialize(n1,n2,n3):
        def HeWeightInit (nprev,ncur):
            std = sqrt(2.0 / nprev)
            # generate random numbers
            weights = np.random.standard_normal(size=(ncur,nprev))
            # scale to the desired range
            scaled_weights = weights * std
            return scaled_weights
        # n1 nodes of input layer
        # n2 nodes of hidden layer
        # n3 nodes of output layer
        W1init = HeWeightInit(n1,n2) 
        B1init = np.random.uniform(-0.5,0.5,size=(n2,1))  
        W2init = HeWeightInit(n2,n3) 
        B2init = np.random.uniform(-0.5,0.5,size=(n3,1))  
        return W1init,B1init,W2init,B2init

    def stochastic_gradient_descent(A0,Y,n1,n2,n3,alpha,W1,B1,W2,B2):
        #stochastic gradient descent
        A1,A2,Z1,Z2 = forward_propagation(A0,W1,B1,W2,B2)
        dW1,dB1,dW2,dB2 = back_propagation(A0,A1,A2,Y,W2,Z1,Z2)
        W1,B1,W2,B2 = update(dW1,dW2,dB1,dB2,W1,B1,W2,B2,alpha)        
        return W1,B1,W2,B2,A2
    
    W1,B1,W2,B2 = initialize(n1,n2,n3)
    examples = X.shape[1]
    sum_se = 0
    for i in range(iterations):
        for j in range(examples):
            # convert numpy vector to column vector
            # (n1,) -> (n1,1) to avoid error
            A0 = np.reshape(X[:,j], (n1, 1)) 
            Yj = Y[j]
            W1,B1,W2,B2,A2 = stochastic_gradient_descent(A0,Yj,n1,n2,n3,alpha,W1,B1,W2,B2)
            se = squared_error(A2,Yj)
            sum_se+=se.item()  #convert np array to scalar value
        mse = sum_se/examples
        sum_se = 0
        if (i+1) % 10 == 0:
            print(f'Mean squarred error is :{mse:.3f} at {i+1} epochs') 
            print(f'with {Yavg:.1f} mean fuel consumption')
    
    return W1,B1,W2,B2


pathANN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(pathANN, 'PREPROCESSING')

current_path = os.path.dirname(os.path.abspath(__file__))

X = pd.read_csv(os.path.join(path, 'preprocessed_x_train.csv')).values
Y = pd.read_csv(os.path.join(path, 'preprocessed_y_train.csv')).values

n1,_ = X.shape
n2 = 100
n3 = 1

print('--Model Training--')
W1,B1,W2,B2 = TRAIN_ANN(X,Y,n1,n2,n3)

with open(os.path.join(current_path, "trained_parameters_fuel_consumption.pkl"),"wb") as dump_file:
    pickle.dump((W1, B1, W2, B2),dump_file)



