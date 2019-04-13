##
# Implementation of a polynomial mutiple regression
# Author: Erick García Ramírez
# MCIC-UNAM 2019-2
# If param deg = 1, a standard multpiple linear regression is performed
##
import numpy as np
from custom_routines import custom_routines as cr

class polynomial_linreg:
    a = [] 
    data = []
    deg = 0
    responses = []

    def __init__(self,data,responses,deg): 
            self.data = data
            self.responses = responses
            self.deg = deg

    # Training function
    # @param data: matrix of training set
    #        y: vector of responses for training set        
    #        deg: degree of polynomials to be used in the model, = 1 for multivariate linear regression
    def fit(self):
        M = cr.modelMatrix(self.data,self.deg)
        MT = np.transpose(M)
        self.a = np.dot(np.dot(cr.inv(np.dot(MT,M)),MT), self.responses)

    #def fit(self):
    #    q, r = np.linalg.qr(self.modelMatrix())
    #    self.a = np.dot(np.dot(np.linalg.inv(r),np.transpose(q)), self.responses)
        
    # unstable
    #def fit(self):
    #    X = polynomial_linreg.modelMatrix(self)
    #    XT = np.transpose(X)
    #    I = np.linalg.inv(np.dot(XT,X))
    #    self.a = np.dot(np.dot(I,XT),self.responses)

    # Build model matrix with polynomials of degree deg as basis functions
    def modelMatrix(self):
        n = len(self.data)
        m = len(self.data[0])
        l = m*self.deg + 1
    
        P = np.ndarray(shape=(n,l))
        for i in range(0,n):
            P[i][0] = 1
        for i in range(0,n):
            for j in range(0,m):
                for k in range(1,self.deg + 1):
                    P[i][j*self.deg + k] = self.data[i][j]**k
        return P            
    
    # Return the estimated parameters
    def coef(self): return self.a
    
    def intercept(self): return self.a[0]
    
    def expand(self,x):
        l = len(self.a)
        m = int((l - 1) /self.deg) 
        ex = np.ones(l)
        for j in range(0,m):
            for k in range(1,self.deg + 1):
                ex[j*self.deg + k] = x[j]**k
                #ex[j*self.deg + k] = x[j*self.deg + k-1]**k
        return ex

    def predict(self,x): 
        return np.dot(self.expand(x),self.a)

    # Función suma de errores cuadrados
    def sqerror(self,x,y):
        s = 0.0
        for i in range(0,len(x)):
            s += (self.predict(x[i])-y[i][0])**2
        return np.sqrt(s/2)[0]

    # Función de error cuadrático medio
    def msqerror(self,x,y):
        s = 0.0
        for i in range(0,len(x)):
            s += (self.predict(x[i])-y[i][0])**2
        return np.sqrt(s/len(x))[0]
