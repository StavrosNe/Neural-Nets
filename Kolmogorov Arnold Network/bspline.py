import numpy as np

class BSpline:
    def __init__(self, degree:int = None, 
                 intervals:int = None,
                 input:np.ndarray = None):
        
        """
        Let G be the number of intervals in the original vector.
        The knot vector is augmented as the first and last knots 
        are repeated k+1 times for a clamped spline.
        For example original knot vector: [0,1,2,3]
        -> knot_vector = [0,0,0,0,1,2,3,3,3,3]
        """
        
        self.max_degree = degree
        self.intervals = intervals
        self.imax = self.intervals + self.max_degree -1
        self.X = input
        assert self.X.ndim == 1
        self.knot_vector = self.make_knot_vector()

    def basis_function(self,x, i, p, knot_vector)->np.ndarray:
        # im applying the Cox deBoor reqursion formula for bspline basis functions
        # p is the degree of the basis function
        V = knot_vector
        x = np.asarray(x)  
        
        if p == 0:
            if i !=(self.imax-1):
                return np.where((V[i] <= x) & (x < V[i+1]), 1, 0)
            else:
                return np.where((V[i] <= x) & (x <= V[i+1]), 1, 0)
        
        else:
            # Avoid division by zero error
            if V[i+p] == V[i]:
                A1 = 0
            else:
                A1 = (x - V[i]) / (V[i+p] - V[i])
                
            if V[i+p+1] == V[i+1]:
                A2 = 0
            else:
                A2 = (V[i+p+1] - x) / (V[i+p+1] - V[i+1])
            
            B1 = self.basis_function(x, i, p-1, knot_vector)
            B2 = self.basis_function(x, i+1, p-1, knot_vector)
            
            N = A1 * B1 + A2 * B2

        # evaluates Ni,p at all x data points
        return N
    
    def basis_function_derivative(self,x, i, p, knot_vector)->np.ndarray:

        V = knot_vector
        x = np.asarray(x)  
        
        if p == 0:
            return np.zeros_like(x)
        else:
            # Avoid division by zero error
            if V[i+p] == V[i]:
                A1d = 0
                A1 = 0
            else:
                A1d = 1/(V[i+p] - V[i])
                A1 = (x - V[i]) / (V[i+p] - V[i])
                
            if V[i+p+1] == V[i+1]:
                A2d = 0
                A2 = 0
            else:
                A2d = (-1)/(V[i+p+1] - V[i+1])
                A2 = (V[i+p+1] - x) / (V[i+p+1] - V[i+1])
            
            B1 = self.basis_function(x, i, p-1, knot_vector)
            B2 = self.basis_function(x, i+1, p-1, knot_vector)
            
            B1d = self.basis_function_derivative(x, i, p-1, knot_vector)
            B2d = self.basis_function_derivative(x, i+1, p-1, knot_vector)


            Nd = A1d * B1 + A1 * B1d+ A2d * B2 + A2 * B2d

            return Nd
        
    def evaluate(self,C:np.ndarray = None)->np.ndarray:
        """
        X is a 1d vector
        """
        imax = self.imax
        assert len(C) == imax
        N = []
        try:
            for i in range(imax):  
                B = self.basis_function(self.X,i,self.max_degree,self.knot_vector)
                Ni = B.reshape((len(B), 1))
                N.append(Ni)

        except AssertionError as error:
            pass

        N = np.hstack(N)

        Y_hat = np.dot(N,C)

        return N,Y_hat
    
    def derivative_evaluate(self,C:np.ndarray = None)->np.ndarray:
        """
        X is a 1d vector
        """
        imax = self.imax
        assert len(C) == imax
        

        N_der = []
        try:
            for i in range(imax):  
                B = self.basis_function_derivative(self.X,i,self.max_degree,self.knot_vector)
                Ni = B.reshape((len(B), 1))
                N_der.append(Ni)

        except AssertionError as error:
            pass

        N_der = np.hstack(N_der)

        Y_hat_der = np.dot(N_der,C)

        return Y_hat_der
        
    def make_knot_vector(self)->np.ndarray:
        """
        creates a clamped knot vector, 
        depending on the input min and max value.
        X has shape (m,1),
        where m is the number of examples
        """
        def k_v_create(tmin,tmax, degree:int = None, intervals:int = None ):
            h = 1e-5
            k = degree
            vector = np.linspace(tmin-h,tmax+h,intervals)
            prev_int = np.full(k, tmin-h)
            after_int = np.full(k, tmax+h)

            knot_vector = np.concatenate((prev_int, vector, after_int))
            return knot_vector
        
        def find_domain(X):
            min_value = np.min(X)
            max_value = np.max(X)
    
            return min_value,max_value
        
        min_value,max_value = find_domain(self.X)

        knot_vector = k_v_create(min_value,max_value,degree=self.max_degree,
                                 intervals=self.intervals)
    

        return knot_vector