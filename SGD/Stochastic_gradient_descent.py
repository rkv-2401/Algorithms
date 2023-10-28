'''
    A program to implement the Stochastic Gradient Descent algorithm by hand.
    CLASS USAGE - 
    Beta -> Vector of targets, i.e. error/loss at current point of iteration
    Beta_hat -> Ideal loss, the lowest possible error for this system of matrices when solved by Ridge Regression. 
'''
import sys
import numpy as np

class SGD_regressor():

    def __init__(self):
        self.fit_intercept_ = False
        self.beta_hat_ = 0
        self.beta_zero_ = 0
        self.iters_ = 0
        self.intercept_ = 0
        self.Id_shape_ = 0
        self.phi_ = 0

    def __repr__(self):
        print("Custom SGD regressor object. \n")
        print("""
                Attributes:
                        beta_hat_ -> Lowest possible error for this system of matrices. Same as SciKitLearn's .coef_ attribute.
                        iters_ -> A tuple containing alpha value and number of iterations taken to fit the regressor to this system, if beta_hat has been reached.
                        intercept_ -> Independent term in decision function. Not used if the data is centered. Set to false by default.
                        Id_shape_ -> Shape of beta vector we are trying to minimize.
                        phi_ -> Regularization coefficient. Set to 0.5 by default.
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
        Xt = np.transpose(X)
        inverse_term =  np.linalg.inv ((1/n) * np.matmul(Xt,X) + phi * np.eye(self.Id_shape_))
        XtY =  np.matmul(Xt,Y)
        beta_hat = (1/n) * np.matmul(inverse_term, XtY)
        print("Beta-hat : ", beta_hat)
        self.beta_hat_ = beta_hat
        self.beta_zero_ = np.ones(self.Id_shape_)
    
    # Assert this is equal to SKLearn's beta_hat estimator.
    
    # This could be a staticmethod if we also pass in phi as an argument.
    @staticmethod
    def SGD_loss(X, Y, phi, beta):
        n = Y.shape[0]
        inverse_term = np.linalg.norm(Y - np.matmul(X, beta))
        loss = (1/n) * pow(inverse_term, 2) + phi * (pow(np.linalg.norm(beta), 2))
        #print("SGD loss : ", loss)
        return loss

    def SGD_grad(self, X, Y, alpha, epochs):
        # Don't need betas history
        sgd_deltas_history = []
        beta = self.beta_zero_
        beta_hat = self.beta_hat_
        phi = self.phi_
        max_index = X.shape[0]
        if (self.fit_intercept_):
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
        self.phi_ = phi
        self.fit_intercept_ = fit_intercept
        # First, find the ideal error.
        self.Id_shape_ = X.shape[1]
        self.beta_hat(self, X, Y, phi)
        # Attribute member beta_zeros_ has been set now.
        # Beta can be safely modified, but beta_zero cannot.
        beta, deltas_history = self.SGD_grad(X, Y, alpha, epochs)
        print("\n Current step size: ", alpha)
        print("Predicted error after", alpha," iterations: ", beta)

regressor = SGD_regressor()
potential_alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]

# To do - loop through all alphas without creating separate objects for each regressor
alpha = potential_alphas[-3]

# Now to put the dataset itself here. Center it, etc.

regressor.fit(X, Y, alpha, epochs = 5)