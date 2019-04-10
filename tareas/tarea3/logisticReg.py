##
# Implementation of logistic  regression
# Author: Erick García Ramírez
# MCIC-UNAM 2019-2
##
import numpy as np
from custom_routines import custom_routines as cr

class logistic_regression:
    coefs = [] 
    minerror = 10.0
    features = []
    responses = []

    #def __init__(self, features, responses): 
    #        self.features = features
    #        self.responses = responses

    def fit(self, features, responses):
        self.features = features
        self.responses = responses
        theta0 = np.zeros(len(features[0]))
        alpha = 0.1
        error = 0.00001
        opt = cr.gd_minimize(self.E, self.gdE, theta0, alpha, error)
        self.coefs = opt[0]
        self.minerror = opt[1]

    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))

    def pi(self, theta,x):
        return self.sigmoid(np.dot(theta,x))

    def p(self, theta):
        arr = np.zeros(len(self.responses))
        for i in range(0,len(self.features)):
            arr[i] = self.pi(theta,self.features[i])
        return arr

    def gdE(self, theta): 
        return np.dot(np.transpose(self.features),self.p(theta)-self.responses)

    def E(self,theta):
        e = 0.0
        for i in range(0,len(self.responses)):
            pi = self.pi(theta, self.features[i])
            yi = self.responses[i]
            e += yi * np.log(pi) + (1-yi) * np.log(1-pi)
        return -e
    
    # Return the estimated parameters
    def coef(self): return self.coefs
    
    def intercept(self): return self.coefs[0]
    
    def predict(self,X): 
        predicted = np.array(len(X))
        for i in range(0,len(X)):
            predicted[i] = np.dot(x, self.coefs)
        return predicted
