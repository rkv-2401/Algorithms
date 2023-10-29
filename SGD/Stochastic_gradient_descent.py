'''
    A program to implement the Stochastic Gradient Descent algorithm by hand.
    DEFINITIONS - 
    Beta -> Vector of targets, i.e. error/loss at current point of iteration
    Beta_hat -> Ideal loss, the lowest possible error for this system of matrices when solved by Ridge Regression.
    ..to be contd. 
'''
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SGD_regressor():

    def __init__(self):
        self.fit_intercept = False
        self.beta_hat_ = 0
        self.beta_zero = 0
        self.iters = 0
        self.intercept = 0
        self.Id_shape = 0
        self.phi = 0

    def __repr__(self):
        print("Custom SGD regressor object. \n")
        print("""
                Attributes:
                        beta_hat_ -> Lowest possible error for this system of matrices. Same as SciKitLearn's .coef_ attribute.
                        iters -> A tuple containing alpha value and number of iterations taken to fit the regressor to this system, if beta_hat has been reached.
                        intercept -> Independent term in decision function. Not used if the data is centered. Set to false by default.
                        Id_shape -> Shape of beta vector we are trying to minimize.
                        phi -> Regularization coefficient. Set to 0.5 by default.
                Methods: 
                        beta_hat(X, Y, phi) -> Used to determine beta_hat value and set the class attribute.
                        _SGD_loss() -> Calculates the loss/error at each step of execution.
              """)

    # This function determines the closed inverse form of beta-hat (the lowest possible error -> inherent bias)
    def beta_hat(self, X, Y, phi):
        '''
            Arguments: X matrix
            Y matrix 
            phi -> 

            Return value: The lowest possible error for this system of matrices.

            Once SGD_regressor object is fit on a system of matrices, update the instance's beta_hat_ attribute member.
        '''
        n = Y.shape[0]
        Xt = np.transpose(X)
        inverse_term =  np.linalg.inv ((1/n) * np.matmul(Xt,X) + phi * np.eye(self.Id_shape))
        XtY =  np.matmul(Xt,Y)
        beta_hat = (1/n) * np.matmul(inverse_term, XtY)
        self.beta_hat_ = beta_hat
        self.beta_zero = np.ones(self.Id_shape)
    
    # Assert this (self.beta_hat_) is equal to SKLearn's beta_hat estimator.
    
    @staticmethod
    def SGD_loss(X, Y, phi, beta):
        n = Y.shape[0]
        inverse_term = np.linalg.norm(Y - np.matmul(X, beta))
        loss = (1/n) * pow(inverse_term, 2) + phi * (pow(np.linalg.norm(beta), 2))
        #print("SGD loss : ", loss)
        return loss

    def SGD_grad(self, X, Y, alpha, epochs):
        # Unlike regular gradient descent, we don't need betas history
        sgd_deltas_history = []
        beta = self.beta_zero
        beta_hat = self.beta_hat_
        phi = self.phi
        max_index = X.shape[0]
        if (self.fit_intercept):
            # This is a fair amount of work with centering the data during the fitting process. Todo.
            print("This feature hasn't been implemented yet.")
        else:
            # Loop through epochs
            for _ in range(0, epochs):
                # Loop through iterations until the index is reached.
                for _ in range(max_index):
                    random_index = np.random.randint(max_index)
                    # list within list problem
                    xi = X[random_index : random_index + 1][0]
                    yi = Y[random_index : random_index + 1][0]
                    gradient = 2 * (xi @ beta - yi) * xi + 2 * phi * beta
                    beta = beta - alpha * gradient
                    delta = SGD_regressor.SGD_loss(X, Y, phi, beta) - SGD_regressor.SGD_loss(X, Y, phi, beta_hat)
                    sgd_deltas_history.append(delta)
        return beta, sgd_deltas_history
            


    def fit(self, X, Y, alpha, phi = 0.5, fit_intercept = False, epochs = 5):
        '''
            Arguments: 
            X -> The matrix derived from the features of the training data, after doing all the requisite preprocessing.
            Y -> The matrix derived from the results of the training data.
            alpha -> The step size used for the regression.
            beta_zero -> The target vector of losses that we aim to minimize.
            fit_intercept -> True or False.
            epochs -> The number of epochs the iteration should step through, if the beta_hat (ideal result) has not been achieved yet.
        '''
        self.phi = phi
        self.fit_intercept = fit_intercept
        # First, find the ideal error.
        self.Id_shape = X.shape[1]
        self.beta_hat(self, X, Y, phi)
        print("Predicted beta-hat (lowest possible error via closed form solution) : ", self.beta_hat_)
        # Attribute member beta_zeros_ has been set now.
        # Beta can be safely modified, but beta_zero cannot.
        beta, deltas_history = self.SGD_grad(X, Y, alpha, epochs)
        print("\n Current step size: ", alpha)
        print("Predicted error after", alpha," iterations: ", beta)

regressor = SGD_regressor()
potential_alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]

# To do - loop through all alphas without creating separate objects for each regressor

###
# Now to put the dataset itself here. Center it, etc.
# This will eventually be in a different function or file, because it's not part of the library itself..

# Part of test case - check that loss goes down as expected with increase in alpha
alpha = potential_alphas[-3]

carSeats = pd.read_csv('CarSeats.csv')

catCols = carSeats.select_dtypes("object").columns
carSeats.drop(columns = catCols, inplace = True)

# This is the target variable, 'Y'.
targetCol = carSeats['Sales']
numCols = carSeats.select_dtypes("number").columns

# Deep copy
trainCols = carSeats.copy()
Y = trainCols['Sales']
x = trainCols.drop(columns = "Sales")

scaler = StandardScaler()
scaler.fit(x)
scaledX = scaler.transform(x)
# This is centering the dataset.
Y = Y.mean()

print(f"Means of features : {scaledX.mean(axis=0)}")
print(f"Variances of all features : {scaledX.var(axis=0)}")

X_train, X_test, y_train, y_test = train_test_split(scaledX, Y, test_size = 0.5, shuffle = False)

regressor.fit(X_train, y_train, alpha, epochs = 5)