##
# Common routines
# Author: Erick García Ramírez
# MCIC-UNAM 2019-2
##
import numpy as np
from scipy.optimize import minimize

class custom_routines:

    #def __init__(self,data,responses,deg): 
    #        self.data = data
    #        self.responses = responses
    #        self.deg = deg

    # Inverting a matrix through QR decomposition
    def inv(M):
        Q, R = np.linalg.qr(M)
        return np.dot(np.linalg.inv(R),np.transpose(Q))
        
    # Build model matrix with polynomials of degree deg as basis functions
    def modelMatrix(M,deg):
        n = len(M)
        m = len(M[0])
        l = m*deg + 1
    
        P = np.ndarray(shape=(n,l))
        for i in range(0,n):
            P[i][0] = 1
        for i in range(0,n):
            for j in range(0,m):
                for k in range(1,deg + 1):
                    P[i][j*deg + k] = M[i][j]**k
        return P            
    
    def sqerror(M,y):
        if len(M) != len(y):
            print("Arrays of incompatible length")
            return 0
        else:
            error = 0.0
            for i in range(0,len(M)):
                error += (M[i]-y[i])**2
        return np.log(error/len(y))
    
    # Minise a function by the gradient descend algorithm
    # @param fnc function to minimise
    #        gdf espression for the gradient of fnc
    #        theta0 starting point of the algorithm, use 0 as default
    #        alpha the learning rate
    #        error the permissible error, use 10^-7 as defualt
    def gd_minimize(fnc, gdf, theta0, alpha, error):
        max_it = 1000000
        thetamin = theta0
        temp = thetamin
        e = 10.0
        iter = 0

        # Find minimum iteratively by gradient descend
        while error < e and iter < max_it:
            temp = thetamin
            thetamin = thetamin - (alpha * gdf(thetamin))
            iter +=1
            e = np.abs(fnc(thetamin) - fnc(temp)) 
        return list(thetamin), fnc(thetamin) 

#    def gd_minimize(fnc, gdf, theta0, alpha):
 #       return gd_minimize(fnc, gdf, theta0, alpha, 0.0000001)
