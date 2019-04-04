##
# Common routines
# Author: Erick García Ramírez
# MCIC-UNAM 2019-2
##
import numpy as np

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
        return np.log(error/2)
