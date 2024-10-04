import numpy as np
import matplotlib.pyplot as plt
from bspline import BSpline

class Simple_Kan_hidden_layer_1():
    def __init__(self, degree:int = None, input:np.ndarray = None,
                 intervals:int = None 
                 ):
    
        self.max_degree = degree
        self.intervals = intervals
        self.imax = self.intervals + self.max_degree -1
        self.X0 = input
        shape = self.X0.shape
        self.m = shape[0]
        self.n = shape[1]

    def init(self):
        self.spline_init()
        self.init_control_points()

    def forward_propagation(self):
        self.spline_evaluate()
        self.evaluate_layer()
        return self.Xmid

    def spline_init(self):
        Splines = np.empty((self.n, 2*self.n+1),dtype=object)
        for i in range(self.n):
            for j in range(2*self.n+1):
                Splines[i][j] = BSpline(degree=self.max_degree, 
                                        intervals = self.intervals,
                                        input = self.X0[:,i])
        self.Splines = Splines
    
    def init_control_points(self):
        
        C = np.empty((self.n, 2*self.n+1),dtype=object)
        for i in range(self.n):
            for j in range(2*self.n+1):
                c = np.random.normal(0, 1, self.imax)
                C[i][j] = c
    
        self.C = np.array([[C[i][j] for j in range(2*self.n+1)] for i in range(self.n)])
        

    def spline_evaluate(self):
        Spline_Values = np.empty((self.n, 2*self.n+1),dtype=object)
        Basis = np.empty((self.n, 2*self.n+1),dtype=object)
        for i in range(self.n):
            for j in range(2*self.n+1):
                Bspline_instance = self.Splines[i][j] #instance of the BSpline class
                basis,values = Bspline_instance.evaluate(C = self.C[i][j])
                Spline_Values[i][j] = values
                Basis[i][j] = basis

        self.Basis_tensor = np.array([[Basis[i][j] for j in range(2*self.n+1)] for i in range(self.n)])
        self.Spline_Values_tensor = np.array([[Spline_Values[i][j] for j in range(2*self.n+1)] for i in range(self.n)])

        
    def evaluate_layer(self):
        X_mid = []
        for j in range(2*self.n+1):
            stacked = np.stack(self.Spline_Values_tensor[:,j])
            x = np.sum(stacked, axis=0)
            X_mid.append(x)

        self.Xmid = np.transpose(np.array(X_mid))
    
    def back_propagation(self,dldx:np.ndarray=None, a=None):
        dX_ex = np.tile(np.expand_dims(np.transpose(dldx), axis=0), (self.n, 1, 1))
        dldC = (dX_ex[:, :, :, np.newaxis] * self.Basis_tensor).sum(axis=2)
        dC = dldC

        assert dC.shape == self.C.shape

        self.C += -a* dC

class Simple_Kan_hidden_layer_2():
    def __init__(self, degree:int = None, input:np.ndarray = None,
                 intervals:int = None 
                 ):
    
        self.max_degree = degree
        self.intervals = intervals
        self.imax = self.intervals + self.max_degree -1
        self.X0 = input
        shape = self.X0.shape
        self.m = shape[0]
        self.n = shape[1]
    
    def init(self):
        self.spline_init()
        self.init_control_points()
    
    def update(self,X:np.ndarray):
        self.X0 = X
        self.spline_update()

    
    def forward_propagation(self):
        self.spline_evaluate()
        self.evaluate_layer()
        return self.Y_hat

    def spline_init(self):
        Splines = np.empty(self.n,dtype=object)
        for i in range(self.n):
            Splines[i] = BSpline(degree=self.max_degree, 
                                intervals = self.intervals,
                                input = self.X0[:,i])
        self.Splines = Splines

    def spline_update(self):
        Splines = np.empty(self.n,dtype=object)
        for i in range(self.n):
            Splines[i] = BSpline(degree=self.max_degree, 
                                intervals = self.intervals,
                                input = self.X0[:,i])
        self.Splines = Splines

    def init_control_points(self)->np.ndarray:
        C = np.empty(self.n,dtype=object)
        for i in range(self.n):
            c = np.random.normal(0, 1, self.imax) 
            C[i] = c
        
        self.C = np.stack(np.array(C))
    
    def spline_evaluate(self):
        Spline_Values = np.empty(self.n,dtype=object)
        Basis = np.empty(self.n,dtype=object)
        for i in range(self.n):
            Bspline_instance = self.Splines[i] 
            basis,values = Bspline_instance.evaluate(C = self.C[i])
            Spline_Values[i] = values
            Basis[i] = basis

        self.Basis_tensor = np.stack(np.array(Basis))
        self.Spline_Values_tensor = np.stack(np.array(Spline_Values))


    def evaluate_layer(self):
        stacked = np.stack(self.Spline_Values_tensor)
        Y_hat = np.sum(stacked, axis=0)
        self.Y_hat = np.transpose(Y_hat)
    
    def spline_evaluate_derivative(self):
        Spline_Derivatives = np.empty(self.n,dtype=object)
        for i in range(self.n):
            Bspline_instance =self.Splines[i] 
            derivative = Bspline_instance.derivative_evaluate(C = self.C[i])
            Spline_Derivatives[i] = derivative

        self.Splines_Derivatives_tensor = np.transpose(np.stack(np.array(Spline_Derivatives)))
    
    def back_propagation(self,Y:np.ndarray=None, a=None):

        self.spline_evaluate_derivative()
        Y_hat = self.Y_hat.reshape((len(self.Y_hat)),1)
        Y = Y.reshape((len(Y)),1)
        dY = Y_hat - Y  
        dldy = np.transpose((1/self.m)*dY) 
        dldh = np.tile(dldy, (self.n, 1))  
        dldC = np.tensordot(dldy, self.Basis_tensor, axes=([1], [1]))
        dC = np.squeeze(dldC) 
        dldXmid = self.Splines_Derivatives_tensor*np.transpose(dldh)

        assert dC.shape == self.C.shape

        self.C += -a* dC
        return dldXmid


class Simple_Kan():
    def __init__(self, max_degree:int = None, input:np.ndarray = None,
                 intervals:int = None , target:np.ndarray = None
                 ):

        self.Y = target
        self.intervals = intervals
        self.X0 = input
        self.layer_1 = Simple_Kan_hidden_layer_1(degree=max_degree,input=self.X0,intervals=intervals)
        self.layer_1.init()
        X_mid = self.layer_1.forward_propagation()
        self.layer_2 = Simple_Kan_hidden_layer_2(degree=max_degree,input=X_mid,intervals=intervals)
        self.layer_2.init()

    def loss_function(self)->np.array:
        mse = 0.5*np.mean((self.Y_hat - self.Y) ** 2)
        return mse

    def train(self,epochs:int = None, learning_rate = None, show_progress:bool =True):
        for i in range(1,epochs+1):
            X_mid = self.layer_1.forward_propagation()
            self.layer_2.update(X_mid)
            self.Y_hat = self.layer_2.forward_propagation()
            dldXmid = self.layer_2.back_propagation(Y = self.Y, a = learning_rate)
            self.layer_1.back_propagation(dldx = dldXmid, a = learning_rate)

            if (i%100 == 0 or i == 1) and show_progress == True:
                mse = self.loss_function()
                print(f'Epoch {i}/{epochs}:{mse}')
    
    def forward(self,X:np.ndarray)->np.ndarray:
        X_mid = self.layer_1.forward_propagation()
        self.layer_2.update(X_mid)
        Y_hat = self.layer_2.forward_propagation()

        return Y_hat

