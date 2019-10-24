import numpy as np

from features_engineering import augment

######################################
# Helper Functions
######################################

def sigmoid(t):
    """applies the sigmoid function to t."""
    return np.exp(t)/(np.exp(t)+1)

def random_batches(y, tx, num_batches):
    """generates num_batches random batches of size batch_size"""
    data_size = len(y)
    shuffle_indices = np.random.permutation(np.mod(np.arange(num_batches),data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    reuturn zip(shuffled_y,shuffled_tx)

######################################
# Loss Functions
######################################

def log_likelihood_loss(y, tx, w):
    """computes the cost by negative log likelihood."""
    p_1 = sigmoid(tx@w)
    p_0 = np.log(1-p_1)
    p_1 = np.log(p_1)
    return -np.sum((y == 1)*p_1+(y == 0)*p_0)

def compute_cost(y, tx, w, method="mae"):
    """computes the cost by mse or mae"""
    err = y - tx.dot(w)
    if method.lower() == "mae":
        cost_f = np.mean(np.abs(err))
    elif method.lower() == "mse":
        cost_f = np.mean(err**2)/2
    else:
        return NotImplementedError
    return cost_f

######################################
# Gradient Functions
######################################

def compute_gradient(y, tx, w, method="mae"):
    """computes the gradient of the mse or mae"""
    err = y - tx.dot(w)
    if method.lower() == "mse":
        grad = -tx.T.dot(err) / len(err)
    elif method.lower() == "mae":
        sign=(err<0)*(-1)+(err>=0)*1
        sign=np.reshape(sign,(-1,1))
        grad=np.sum(tx*sign, axis=0)*(-1)/len(err)
    else: 
        return NotImplementedError
    return grad

def log_likelihood_gradient(y, tx, w):
    """computes the gradient of the log likelihood."""
    return tx.T@(sigmoid(tx@w)-y)

######################################
# Least Squares
######################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """applies least squares using gradient descent to optimize w"""
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = compute_gradient(y, tx, w)
        # gradient w by descent update
        w -= gamma * grad

    return w, compute_cost(y, tx, w)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """applies least squares using stochastic gradient descent to optimize w"""
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_gradient(y, tx, initial_w)
    for n_iter in range(max_iters):
        for yb, txb in random_batches(y, tx, max_iters):
            # compute 1 SGD and the loss
            grad = compute_gradient(yb, txb)
            # update w
            w -= gamma * grad

    return w, compute_cost(y, tx, w)

def least_squares(y, tx):
    """applies pure least squares to optimize w"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_cost(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """applies ridge regression to optimize w"""
    lambdaI = (lambda_ * 2 * len(y)) * np.eye(tx.shape[1])
    a = (tx.T.dot(tx)) + lambdaI
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_cost(y, tx, w)
    return w, loss

################################################
# Logistic Regression
################################################


def logistic_regression_GD(y, tx, initial_w, max_iter, gamma):
    """applies logistic regression using gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # logistic regression
    for iter in range(max_iter):
        # updating the weights
        grad = log_likelihood_gradient(y, tx, w)
        w -= gamma*grad

    return w, log_likelihood_loss(y, tx, w)


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma):
    """applies logistic regression using stochastic gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # logistic regression
    for yb, txb in random_batches(y, tx, max_iters):
        # updating the weights
        grad = log_likelihood_gradient(yb, txb, w)
        w -= gamma*grad

    return w, log_likelihood_loss(y, tx, w)

##################################################
# Regularized Logistic Regression
##################################################


def reg_logistic_regression_GD(y, tx, lambda_, initial_w, max_iters, gamma):
    """applies regularized logistic regression using gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # regularized logistic regression
    for iter in range(max_iters):
        # updating the weights
        grad = log_likelihood_gradient(y, tx, w)+2*lambda_*w
        w -= gamma*grad

    loss = log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T@w)
    return w, loss


def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    """applies regularized logistic regression using stochastic gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # regularized logistic regression
    for yb, txb in random_batches(y, tx, max_iters):
        # updating the weights
        grad = log_likelihood_gradient(yb, txb, w)+2*lambda_*w
        w -= gamma*grad

    loss = log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T@w)
    return w, loss

##################################################
# Cross-Validation
##################################################

def build_k_indices(num_row, k_fold, seed=1):
    """splits indices of data into 'k_folds' folds."""
    # setting the random seed
    np.random.seed(seed)
    
    # interval computation
    interval = num_row // k_fold

    # build k_folds indices
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval:(k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def split_train_test(y, tx, k_indices, k):
    """splits data into train and test subsets given the fold indices 'k_indices' and the fold index 'k'."""
    # get the test split
    test_ind = k_indices[k]

    # get the k-1 train splits
    train_splits = np.delete(k_indices,k,0)
    train_ind = k_indices[train_splits].reshape(-1)

    x_train = tx[train_ind]
    x_test = tx[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]

    return x_train, x_test, y_train, y_test

def cross_validation(y, tx, k_indices, num_folds, lamda, degree, iter=1000, gamma=1e-2, reg_f, loss_f):
    """
    runs cross validation, for every fold splits the data into test and train does feature expansion
    and trains with reg_f, returns overall error.
    """
    # initializing useful variables
    errors = np.zeros(num_folds)
    initial_w = np.zeros(x_train.shape[1])
    
    # feature expansion
    x_train_aug=augment(x_train, degree)
    x_test_aug=augment(x_test, degree)
    
    # for each fold
    for k in range(num_folds):
        x_train, x_test, y_train, y_test = split_train_test(y, tx, k_indices, k)
        w = reg_f(y_train, x_train_aug, lamda, initial_w, iter, gamma)
        errors[k] = loss_f(y_test, x_test_aug, w)
    
    # computing the average error
    avg_error = np.mean(errors)
    return avg_error

def cross_validation_SGD(y, tx, k_indices, num_folds, lamda, degree, iter=1000, gamma=1e-2, reg_f, loss_f):
    """runs cross validation: does feature expansion and trains with reg_f, returns full set error."""
    # initializing useful variables
    initial_w = np.zeros(x_train.shape[1])
    
    # feature expansion
    x_train_aug=augment(x_train, degree)
    x_test_aug=augment(x_test, degree)
    
    w = reg_f(y_train, x_train_aug, lamda, initial_w, iter, gamma)
    err= loss_f(y_test, x_test_aug, w)
    
    # computing the average error
    avg_error = np.mean(errors)
    return avg_error

def find_besthyperparameters_CrossValid(y, tx, num_folds, lamdas, degrees, cross_val_f, reg_f, loss_f):
    """finds the hyperparameters that give the lowest error for the cross-validation"""
    # initializing useful variables
    k_indices = build_k_indices(len(y), num_folds)
    errors = np.zeros([lamdas.shape[0], degrees.shape[0]])
    
    # for each hyperparameter
    for i, lamd in enumerate(lamdas):
        for j, deg in enumerate(degrees):
            errors[i, j] = cross_val_f(y, tx, k_indices, num_folds,lamd,deg,reg_f,loss_f)

    # evaluating which hyperparameters are the best
    degree_best = degrees[np.argmin(errors) % len(degree)]
    lambda_best = lamdas[np.argmin(errors) // len(degree)]

    return lambda_best, degree_best
