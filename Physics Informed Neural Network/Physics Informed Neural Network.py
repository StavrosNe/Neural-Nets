import numpy as np

import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def initialize_adam(parameters):
    v = {}
    s = {}
    
    for key in parameters.keys():
        v[key] = np.zeros_like(parameters[key])
        s[key] = np.zeros_like(parameters[key])
        
    return v, s

def initialize(n1, n2, n3, n4):
    
    W1 = np.random.normal(0,0.1,size = (n2, n1))
    B1 = np.random.uniform(-0.5, 0.5, size=(n2, 1))
    W2 = np.random.normal(0,0.1,size = (n3, n2))
    B2 = np.random.uniform(-0.5, 0.5, size=(n3, 1))
    W3 = np.random.normal(0,0.1,size = (n4, n3))
    B3 = np.random.uniform(-0.5, 0.5, size=(n4, 1))

    parameters = {'W1':W1,'W2':W2,'W3':W3,'B1':B1,'B2':B2,'B3':B3}
    return parameters


def forward_propagation(A0, parameters:dict):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    B1 = parameters['B1']
    B2 = parameters['B2']
    B3 = parameters['B3']

    Z1 = np.dot(W1, A0) + B1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = tanh(Z2)
    Z3 = np.dot(W3, A2) + B3
    Y_hat = Z3 #Y_hat

    forward = {'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2,'Z3':Z3,'Y_hat':Y_hat}
    return forward

def back_propagation(A0,Y,forward:dict,parameters:dict,dydx):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    A1 = forward['A1']
    A2 = forward['A2']
    Y_hat = forward['Y_hat']

    Z1 = forward['Z1'] 
    Z2 = forward['Z2']
    Z3 = forward['Z3']


    network_initial_condtion = forward_propagation(A0[0][0]*np.ones_like(A0), parameters) #netowrk ouput at x = A0[0][0]
    Y0 = network_initial_condtion ['Y_hat']

    
    initial_condition_error = Y0 - Y[0]
    ode_error = dydx - Y_hat 

    l1 = 1
    l2 = 0.1
    
    m = A0.shape[1]

    dldy = (1/m) * (l1*ode_error + l2*initial_condition_error)

    dldz3 = dldy
    dldw3 = np.dot(dldz3, np.transpose(A2))
    dldb3 = np.sum(dldz3,axis=1, keepdims=True)

    dlda2 = np.dot(W3.T, dldz3)
    dldz2 = dlda2 * tanh_derivative(Z2)
    dldw2 = np.dot(dldz2, np.transpose(A1))
    dldb2 = np.sum(dldz2,axis=1, keepdims=True)

    dlda1 = np.dot(W2.T, dldz2)
    dldz1 = dlda1 * tanh_derivative(Z1)
    dldw1 = np.dot(dldz1, np.transpose(A0))
    dldb1 = np.sum(dldz1,axis=1, keepdims=True)


    backward = {'W1':dldw1,'B1':dldb1,'W2':dldw2,'B2':dldb2,'W3':dldw3,'B3':dldb3}

    return backward


def gradient(forward:dict,parameters:dict):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    Z1 = forward['Z1'] 
    Z2 = forward['Z2']
   
    # gradients in relation to network output Y

    dyda2 = W3
    da2dz2 = tanh_derivative(Z2)
    dz2da1 = W2
    da1dz1 = tanh_derivative(Z1)
    dz1dx = W1

    dydx = dydx = np.dot(W3, da2dz2 * np.dot(W2, da1dz1 * W1))

    return dydx



def update(parameters,gradients,learning_rate):

    for key in parameters.keys():
        parameters[key] -= learning_rate * gradients[key]

    return parameters

def update_adam(parameters, grads, v, s, t, learning_rate=None, beta1=0.9, beta2=0.999, epsilon=1e-8):
    v_corrected = {}
    s_corrected = {}
    
    for key in parameters.keys():
        v[key] = beta1 * v[key] + (1 - beta1) * grads[key]
        s[key] = beta2 * s[key] + (1 - beta2) * (grads[key] ** 2)

        v_corrected[key] = v[key] / (1 - beta1 ** t)
        s_corrected[key] = s[key] / (1 - beta2 ** t)

        parameters[key] -= learning_rate * (v_corrected[key] / (np.sqrt(s_corrected[key]) + epsilon))

    return parameters, v, s

def mean_squared_error(Y_hat, Y):
    mse = np.mean((Y_hat - Y) ** 2)
    return mse



def train(A0, Y, epochs, a, n1, n2, n3, n4):

    parameters = initialize(n1, n2, n3, n4)
    v,s = initialize_adam(parameters)

    for i in range(1, epochs + 1):

        forward = forward_propagation(A0, parameters)
        dydx = gradient(forward,parameters)
        gradients = back_propagation(A0,Y,forward,parameters, dydx)
        parameters,v,s = update_adam(parameters, gradients, v, s, i, learning_rate=a)
        #parameters = update(parameters,gradients,a)

        Y_hat = forward['Y_hat'].flatten()

        if i % 100 == 0 or i==1:
            mse = mean_squared_error(Y_hat, Y)
            print(f'Epoch: {i}/{epochs} MSE: {mse}')

    return Y_hat

n1 = 1   #input
n2 = 18  #hidden 1
n3 = 18  #hidden 2
n4 = 1   #output

X = np.linspace(0, 3, 300)

Y = np.exp(X) #ode solution

m = len(X)
A0 = X.reshape((n1, m))

epochs = 2000
a = 0.001


Y_hat = train(A0, Y, epochs, a, n1, n2, n3, n4)




plt.figure(figsize=(12, 8))
plt.plot(X,Y, color='black', linewidth=2)
plt.plot(X,Y_hat, linestyle='--', color='gray', linewidth=2)
plt.title('PINN')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.show()