import numpy as np
import pickle
import os


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def forward_prop(Wx,Ws,Who,b1,bo,e,X):
    xt_3 = X[0]
    xt_2 = X[1]
    xt_1 = X[2]
    xt   = X[3]

    # e is the cutoff error 
    # e = Ws*ht_4 = f(Wx*xt_4+Ws*ht_5)
    # patenta to e de to kserei to chat gpt

    zt_3 = Wx*xt_3
    ht_3 = leaky_relu(zt_3)

    zt_2 = Wx*xt_2 + Ws*ht_3 + b1 + e
    ht_2 = leaky_relu(zt_2)

    zt_1 = Wx*xt_1 + Ws*ht_2 + b1
    ht_1 = leaky_relu(zt_1)

    zt = Wx*xt + Ws*ht_1 + b1
    ht = leaky_relu(zt)

    zy = Who*ht + bo
    Yest = zy #estimation

    return Yest,ht_3,ht_2,ht_1,ht,zt_2,zt_1,zt


def backward_prop(Ws,Who,X,Y,Yest,ht_3,ht_2,ht_1,ht,zt_2,zt_1,zt):

    xt_2 = X[1]
    xt_1 = X[2]
    xt   = X[3]


    dy = Yest - Y
    dzy = 2 * dy

    dht = dzy * Who
    dzt = dht * leaky_relu_derivative(zt)

    dht_1 = dzt * Ws
    dzt_1 = dht_1 * leaky_relu_derivative(zt_1)

    dht_2 = dzt_1 * Ws
    dzt_2 = dht_2 * leaky_relu_derivative(zt_2)


    dWho = dzy * ht
    dbo = dzy

    dWs = dzt * ht_1 + dzt_1 * ht_2 + dzt_2 * ht_3
    dWx = dzt * xt + dzt_1 * xt_1 + dzt_2 * xt_2
    db1 = dzt + dzt_1 + dzt_2
    de = dzt_2

    return dWho,dbo,dWs,dWx,db1,de

def update(Who,bo,Wx,Ws,b1,e,dWho,dbo,dWs,dWx,db1,de,a):
    Who = Who - a * dWho
    bo = bo - a * dbo
    Wx = Wx - a* dWx
    Ws = Ws - a * dWs
    b1 = b1 - a * db1
    e = e - a * de

    return Who,bo,Wx,Ws,b1,e


def sequence(data):
    # start from t == 3 to t == n - 1
    # n = lenght(data) tmax_index = n - 1
    # we want the highest current t to have at least one t+1
    tmax = len(data) - 1

    sequential = []

    targets = []

    for t in range(3,tmax):
        data_sequence = data[t-3:t+1]
        sequential.append(data_sequence)
        targets.append(data[t+1])
    
    return sequential,targets

def loss_function(Yest,Y):
    return (Yest-Y)**2

def training(sequential,targets,a,epochs):
    Wx = 0.1
    Ws = 0.1
    Who = 0.1
    b1 = 0.1
    bo = 0.1
    e = 1
    examples = len(targets)

    for i in range(epochs):

        error = 0

        for X,Y in zip(sequential,targets):

            Yest,ht_3,ht_2,ht_1,ht,zt_2,zt_1,zt = forward_prop(Wx,Ws,Who,b1,bo,e,X)
            dWho,dbo,dWs,dWx,db1,de = backward_prop(Ws,Who,X,Y,Yest,ht_3,ht_2,ht_1,ht,zt_2,zt_1,zt)
            Who,bo,Wx,Ws,b1,e = update(Who,bo,Wx,Ws,b1,e,dWho,dbo,dWs,dWx,db1,de,a)
            error += loss_function(Yest,Y)

        mse = error/examples
        #print(f'Epoch {i} : mse = {mse}')
    return  Who,bo,Wx,Ws,b1,e

t = np.linspace(0, 6*np.pi, 1000)

# Simple signal used for training 
data = np.sin(t) + np.cos(t) + np.random.randint(1)

sequential,targets = sequence(data)

a = 0.001

epochs = 10


Who,bo,Wx,Ws,b1,e = training(sequential,targets,a,epochs)


current_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_path, "trained_parameters_rnn.pkl"),"wb") as dump_file:
    pickle.dump((Who,bo,Wx,Ws,b1,e),dump_file)
