import matplotlib.pyplot as plt
import numpy as np
from simple_kan import Simple_Kan

data_points = 200 
x1 = np.linspace(0,2*np.pi,data_points)
x2 = np.linspace(0,2*np.pi,data_points)
x3 = np.linspace(0,2*np.pi,data_points)
X0 = np.column_stack((x1, x2, x3))

Y = np.sin(x1) + 2*np.cos(2*x2) + 0.5*np.exp(0.1*x1) + x3



model = Simple_Kan(max_degree=3,intervals=3,input=X0,target=Y)
model.train(epochs = 1000,learning_rate = 0.8)
Y_hat = model.forward(X0)


plt.figure(figsize=(12, 8))
plt.plot(Y, label='Y = sin(x1) + 2*cos(2*x2) + 0.5*exp(0.1*x1) + x3', color='green')
plt.plot(Y_hat, label='Simple_Kan',  color='black', linewidth=2)
plt.title('Kolmogorov Arnold Network ')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.legend()
plt.legend()
plt.show()