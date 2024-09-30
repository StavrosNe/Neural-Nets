import matplotlib.pyplot as plt
import numpy as np
from simple_kan import Simple_Kan

data_points = 200 
x1 = np.linspace(0,2*np.pi,data_points)
x2 = np.linspace(0,2*np.pi,data_points)
x3 = np.linspace(0,2*np.pi,data_points)
X0 = np.column_stack((x1, x2, x3))

Y = np.sin(x1)* np.random.uniform(-1, 1, size=200) + 2*np.cos(2*x2) + 0.5*np.exp(0.1*x1) + x3



model = Simple_Kan(max_degree=3,intervals=5,input=X0,target=Y)
model.train(epochs = 1000,learning_rate = 0.8)
Y_hat = model.forward(X0)


plt.figure(figsize=(12, 8))
plt.plot(Y, label='Y = np.sin(x1)* np.random.uniform(-1, 1, size=200) + 2*np.cos(2*x2) + 0.5*np.exp(0.1*x1) + x3', 
         color='black', linewidth=2)
plt.plot(Y_hat, label='Simple_Kan', linestyle='--', color='gray', linewidth=2)
plt.title('Kolmogorov Arnold Network ')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.legend()
plt.legend()
plt.show()