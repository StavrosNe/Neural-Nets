import matplotlib.pyplot as plt
import numpy as np
from simple_kan import Simple_Kan

data_points = 200 
x1 = np.linspace(0,2*np.pi,data_points)
x2 = np.linspace(0,2*np.pi,data_points)
x3 = np.linspace(3,2*np.pi,data_points)
x4 = np.linspace(2,6,data_points)
X0 = np.column_stack((x1, x2, x3))


Y = np.sin(x1*x2) + np.log(5*x4) + np.cos(2*x3) + 0.3*x2**2
model = Simple_Kan(max_degree=3,intervals=6,input=X0,target=Y)
model.train(epochs = 1000,learning_rate = 0.8)
Y_hat = model.forward(X0)

function_label = 'Y(x1,x2,x3,x4) = sin(x1*x2)+log(5*x4)+cos(2*x3)+0.3x2**2'
plt.figure(figsize=(12, 8))
plt.plot(Y, label=function_label, color='black', linewidth=2)
plt.plot(Y_hat, label='Simple_Kan', linestyle='--', color='gray', linewidth=2)
plt.title('Kolmogorov Arnold Network ')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.legend()
plt.legend()
plt.show()