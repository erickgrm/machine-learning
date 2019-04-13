##
# Implementation of logistic regression
# Author: Erick García Ramírez
# MCIC-UNAM 2019-2
##
import numpy as np
from custom_routines import custom_routines as cr

class logistic_regression:
    coefs = [] 
    features = []
    responses = []
    lambd = 0.0

    # Main routine to train the model
    # @param features: matrix of features
    #        responses: vector of corresponding responses
    #        lambd: factor for l2 regularization; if 0, no regularization is applied
    # It is assume features has first column of ones, so the 
    # intercept is estimated directly 
    def fit(self, features, responses, lambd):
        self.features = features
        self.responses = responses
        self.lambd = lambd

        # Fix params for gradient descend algorithm
        theta0 = np.zeros(len(features[0]))
        alpha = 0.1
        error = 0.001

        # Find optimal vector of weights through the gradient descend algorithm
        self.coefs = self.gd_minimize(self.gdE, theta0, alpha, error)


    # Minimise a function by the gradient descend algorithm
    # @param gdf the gradient function
    #        theta0 starting point of the algorithm, default vector of 1's
    #        alpha the learning rate, default 0.1
    #        error the permissible error, default 10^-4 
    def gd_minimize(self, gdf, theta0, alpha, error):
        max_iter = 300 
        #max_iter = 150 
        #max_iter = 1000 
        theta_min = theta0
        temp = theta_min
        e = 1.0
        iter = 0

        # Find minimum iteratively by gradient descend
        while error < e and iter < max_iter:
            temp = theta_min
            theta_min = theta_min - (alpha * gdf(theta_min))
            iter +=1
            e = np.abs(np.dot(np.ones(len(temp)),temp-theta_min))
        return theta_min
    
    # The gradient of the loss function, general to accept l2 regularization with parameter lambd
    def gdE(self, theta): 
        dtheta = np.dot(np.transpose(self.features),self.p(theta)-self.responses)
        return dtheta + (2 * self.lambd * np.dot(np.ones(len(theta)), theta))

    # Gradient without considering penalization at all, the above one is more general
    #def gdE(self, theta): 
    #    return np.dot(np.transpose(self.features),self.p(theta)-self.responses)

    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))

    def p(self, theta):
        arr = np.zeros(len(self.responses))
        for i in range(0,len(self.features)):
            arr[i] = self.sigmoid(np.dot(theta, self.features[i]))
        return arr

    # Return the estimated parameters
    def coef(self): return self.coefs
    
    def intercept(self): return self.coefs[0]
    
    # Return the probabilities of an instance x
    def proba(self, x):
        prob = self.sigmoid(np.dot(self.coefs, x))
        return [1-prob,prob]

    # Predicted classes for an array of instances
    def predict(self,X): 
        predicted = np.zeros(len(X))
        for i in range(0,len(X)):
            if self.proba(X[i])[0] < self.proba(X[i])[1]:
                predicted[i] = 1
        return predicted

    # Predicted probabilities for an array of instances
    def predict_probas(self,X):
        arr = np.zeros(len(X))
        for i in range(0,len(X)):
            arr[i] = self.proba(X[i])
        return  arr

    #
    def decision(self, X):
        arr = np.zeros(len(X))
        for i in range(0,len(X)):
            arr[i] = np.dot(self.coef(), X[i])
        return  arr

    # Score, the proportion of correct predictions on the array X
    def score(self, X, y):
        predicted = self.predict(X)
        return cr.diffcount(predicted, y)/len(y)


    # The loss function -SUM y_i(ln pi) + (1-yi)ln (1-pi)
    # NOT ACTUALLY NEEDED
    def E(self,theta):
        e = 0.0
        for i in range(0,len(self.responses)):
            pi = self.sigmoid(np.dot(theta, self.features[i]))
            yi = self.responses[i]
            e += yi * np.log(pi) + (1-yi) * np.log(1-pi)
        return -e
