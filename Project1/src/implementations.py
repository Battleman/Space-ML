import numpy as np

from features_engineering import augment

######################################
# Helper Functions
######################################


def sigmoid(t):
    """applies the sigmoid function to t."""
    t[t >= 20] = 20
    t[t <= -20] = -20
    return np.exp(t)/(np.exp(t)+1)


def random_batches(y, tx, num_batches):
    """generates num_batches random batches of size batch_size"""
    data_size = len(y)
    shuffle_indices = np.random.permutation(
        np.mod(np.arange(num_batches), data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    return zip(shuffled_y, shuffled_tx)

######################################
# Loss Functions
######################################


def log_likelihood_loss(y, tx, w):
    """computes the cost by negative log likelihood."""
    p_1 = sigmoid(tx.dot(w))
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

def compute_gradient(y, tx, w, method="mse"):
    """computes the gradient of the mse or mae"""
    err = y - tx.dot(w)
    if method.lower() == "mse":
        grad = -tx.T.dot(err) / len(err)
    elif method.lower() == "mae":
        sign = (err < 0)*(-1)+(err >= 0)*1
        sign = np.reshape(sign, (-1, 1))
        grad = (np.sum(tx*sign, axis=0)*(-1)/len(err))[:, np.newaxis]
    else:
        return NotImplementedError
    return grad


def log_likelihood_gradient(y, tx, w):
    """computes the gradient of the log likelihood."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)

######################################
# Least Squares
######################################


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """applies least squares using gradient descent to optimize w"""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad = compute_gradient(y, tx, w)
        # gradient w by descent update
        if n_iter % (max_iters//10) == 0:
            print(compute_cost(y, tx, w))
        w -= gamma * grad

    return w, compute_cost(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """applies least squares using stochastic gradient descent to optimize w"""
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_gradient(y, tx, initial_w)
    for it,(yb, txb) in enumerate(random_batches(y, tx, max_iters)):
            # compute 1 SGD and the loss
            grad = compute_gradient(np.array([yb]), txb[np.newaxis,:], w)
            # update w
            w -= gamma * grad
            if it % (max_iters//10) == 0:
                print(compute_cost(y, tx, w))
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
    #computing the gram matrix
    gram=tx.T@tx
    #diagonalizing the gram matrix
    u, d, ut=np.linalg.svd(gram, full_matrices=True)
    #adding the lmbda matrix to the diagonal matrix to prevent approximation problems
    d+=2*gram.shape[0]*lambda_
    #solving the least squares linear problem
    w=np.linalg.solve(np.diag(d).dot(ut),ut.dot(tx.T.dot(y)))
    return w, compute_cost(y, tx, w)

################################################
# Logistic Regression
################################################


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """applies logistic regression using gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # logistic regression
    for n_iter in range(max_iters):
        # updating the weights
        grad = log_likelihood_gradient(y, tx, w)
        w -= gamma*grad
        if n_iter % (max_iters//10) == 0:
            print(log_likelihood_loss(y, tx, w))
    return w, log_likelihood_loss(y, tx, w)


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma):
    """applies logistic regression using stochastic gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # logistic regression
    for it, (yb, txb) in enumerate(random_batches(y, tx, max_iters)):
        # updating the weights
        grad = log_likelihood_gradient(np.array([yb]), txb[np.newaxis, :], w)
        w -= gamma*grad
        if it % (max_iters//10) == 0:
            print(log_likelihood_loss(y, tx, w))
    return w, log_likelihood_loss(y, tx, w)

##################################################
# Regularized Logistic Regression
##################################################


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """applies regularized logistic regression using gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # regularized logistic regression
    for iter in range(max_iters):
        # updating the weights
        grad = log_likelihood_gradient(y, tx, w)+2*lambda_*w
        #if iter % (max_iters//2) == 0:
            #print(log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T.dot(w)))
        w -= gamma*grad
    loss = log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T.dot(w))
    return w, loss


def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    """applies regularized logistic regression using stochastic gradient descent to optimize w"""
    # initializing the weights
    w = initial_w

    # regularized logistic regression
    for it, (yb, txb) in enumerate(random_batches(y, tx, max_iters)):
        # updating the weights
        grad = log_likelihood_gradient(
            np.array([yb]), txb[np.newaxis, :], w)+2*lambda_*w
        w -= gamma*grad
        #if it % (max_iters//2) == 0:
            #print(log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T.dot(w)))
    loss = log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T.dot(w))
    return w, loss