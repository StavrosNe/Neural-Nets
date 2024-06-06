import numpy as np
from math import sqrt
import pickle
import pandas as pd
import os

pathANN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(pathANN, 'PREPROCESSING')
current_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_path, "trained_parameters_fuel_consumption.pkl"),"rb") as dump_file:
    W1, B1, W2, B2=pickle.load(dump_file)

X = pd.read_csv(os.path.join(path, 'preprocessed_x_evaluate.csv')).values
Y = pd.read_csv(os.path.join(path, 'preprocessed_y_evaluate.csv')).values

def RELU(x):
    # alternative return x * (x > 0)
    return (np.maximum(0,x))


def forward_propagation(A0,W1,B1,W2,B2):
    Z1 = np.dot(W1,A0) + B1
    # Z1 = W1 A0 + B1
    A1 = RELU(Z1)
    Z2 = np.dot(W2,A1) + B2
    # Z2 = W1 A1 + B2
    A2 = RELU(Z2)
    return A2

def squared_error(A2,Y):
    dY = (A2-Y)
    return pow(dY,2)


Yavg = np.mean(Y)
n1,_ = X.shape
examples = X.shape[1]
sum_se = 0
for j in range(examples):
    # convert numpy vector to column vector
    # (n1,) -> (n1,1) to avoid error
    A0 = np.reshape(X[:,j], (n1, 1)) 
    Yj = Y[j]
    A2  = forward_propagation(A0,W1,B1,W2,B2)
    se = squared_error(A2,Yj)
    sum_se+=se.item()  #convert np array to scalar value
mse = sum_se/examples
sum_se = 0

print('--Model Evaluation SGD--')
print('The multi layer perceptron achieved:')
print(f'Mean squarred error is :{mse:.3f}') 
print(f'with {Yavg:.1f} mean fuel consumption')