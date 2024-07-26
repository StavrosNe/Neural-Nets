import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

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

    return Yest


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

t = np.linspace(0, 12*np.pi, 200)



#data_raw = np.cumsum(np.random.randn(len(t))) + np.sin(2*t)

data_raw = 2 + t - t



minimum = min(data_raw)

if minimum < 0:
    data = data_raw - minimum

else:
    data = data_raw

sequential,targets = sequence(data)

current_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_path, "trained_parameters_rnn.pkl"),"rb") as dump_file:
    Who,bo,Wx,Ws,b1,e = pickle.load(dump_file)

print(Who,bo,Wx,Ws,b1,e)

estimation_signal = []

for X in sequential:

    Yest = forward_prop(Wx,Ws,Who,b1,bo,e,X)

    estimation_signal.append(Yest)

tau = t[4:]

plt.figure()
plt.plot(t,data, '-', label='SIGNAL')
plt.plot(tau,  estimation_signal, 'o', label='RNN')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()