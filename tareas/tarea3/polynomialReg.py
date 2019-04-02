class polynomial_linreg:
    a = []

    def modelMatrix(data,deg):
        n = len(data)
        m = len(data[0])
        l = m*deg + 1
    
        P = np.ndarray(shape=(n,l))
        for i in range(0,n):
            P[i][0] = 1
        for i in range(0,n):
            for j in range(0,m):
                for k in range(1,deg):
                    P[i][j+k+1] = data[i][j]**k
        return P
                
    # Main function
    # @param data: matrix of training set
    #        y: vector of responsed for training set        
    #        deg: degree of polynomials to be used in the model, = 1 for multivariate linear regression
    def fit(data,responses,deg):
        X = polynomial_linreg.modelMatrix(data,deg)
        XT = np.transpose(X)
        I = np.linalg.inv(np.dot(XT,X))
        a = np.dot(np.dot(I,XT),responses)
        return a
    
    def coef(): return a
    
    def intercept(): 
        return a[0]
    
    def predict(x): return np.dot(x,a)
