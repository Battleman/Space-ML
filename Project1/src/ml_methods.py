def compute_cost(y, tx, w, method="mae"):
    def calculate_mse(e):
        return np.mean(e**2)/2
    def calculate_mae(e):
        return np.mean(np.abs(e))
    
    if method.lower() == "mae":
        cost_f = compute_mae
    elif method.lower() == "mse":
        cost_f = compute_mse
    else:
        return NotImplementedError
    e = y - tx.dot(w)
    return cost_f(e)


def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, e = compute_gradient(y, tx, w, fn="mse")
        loss = calculate_cost(e)

        # gradient w by descent update
        w -= gamma * grad

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_gradient(y, tx, initial_w)
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute 1 SGD and the loss
            grad, e = compute_gradient(y_batch, tx_batch, w)
            loss = calculate_cost(e)

            # update w
            w -= gamma * grad

    return w, loss


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_cost(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    lambdaI = (lambda_ * 2 * len(y)) * np.eye(tx.shape[1])
    a = (tx.T.dot(tx)) + lambdaI
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_cose(y, tx, w)
    return w, loss

################################################
# Logistic Regression
################################################

def sigmoid(t):
    """applies the sigmoid function to t."""
    return np.exp(t)/(np.exp(t)+1)

def log_likelihood_loss(y, tx, w):
    """computes the cost by negative log likelihood."""
    p_1=sigmoid(tx@w)
    p_0=np.log(1-p_1)
    p_1=np.log(p_1)
    return -np.sum((y==1)*p_1+(y==0)*p_0)

def log_likelihood_gradient(y, tx, w):
    """computes the gradient of the log likelihood."""
    return tx.T@(sigmoid(tx@w)-y)

"""
def random_batches(y, tx, batch_size, num_batches):
    """
    #generates num_batches random batches of size batch_size
    """
    data_size = len(y)
    
    for batch_num in range(num_batches):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
        yield shuffled_y[:batch_size],shuffled_tx[:batch_size]
        
"""
    
def random_batches(y, tx, batch_size, num_batches):
    """
    generates num_batches random batches of size batch_size
    """
    data_size = len(y)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    
    for batch_num in range(min(num_batches,(int)(data_size/batch_size)+1)):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

"""
def random_batches(y, tx, batch_size, num_batches):
    """
    #generates num_batches random batches of size batch_size
    """
    data_size = len(y)
    batches=[]
    batch_cuts=(int)(data_size/batch_size)+1
    
    while True:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
        for batch_cut in range(batch_cuts):
            start_index = batch_cut * batch_size
            end_index = min((batch_cut + 1) * batch_size, data_size)
            if start_index != end_index:
                batches.append([shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]])
            if len(batches)==num_batches:
                return batches
"""
    
def logistic_regression_GD(y, tx, initial_w, max_iter, gamma):
    """
    applies logistic regression using gradient descent to optimize w
    """
    # initializing the weights
    w=initial_w
    
    # logistic regression
    for iter in range(max_iter):
        # updating the weights
        grad=log_likelihood_gradient(y, tx, w)
        w-=gamma*grad
        
    retrun w, log_likelihood_loss(y, tx, w)
    
def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    applies logistic regression using stochastic gradient descent to optimize w
    """
    # initializing the weights
    w=initial_w
    
    # logistic regression
    for yb, txb in random_batches(y, tx, batch_size, max_iters):
        # updating the weights
        grad=log_likelihood_gradient(yb, txb, w)
        w-=gamma*grad
        
    return w, log_likelihood_loss(y, tx, w)

##################################################
# Regularized Logistic Regression
##################################################

def reg_logistic_regression_GD(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    applies regularized logistic regression using gradient descent to optimize w
    """
    # initializing the weights
    w=initial_w
    
    # regularized logistic regression
    for iter in range(max_iters):
        # updating the weights
        grad= log_likelihood_gradient(y, tx, w)+2*lambda_*w
        w-=gamma*grad
        
    loss= log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T@w)
    return w, loss

def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, batch_size, max_iters, gamma):
    """
    applies regularized logistic regression using stochastic gradient descent to optimize w
    """
    # initializing the weights
    w=initial_w
    
    # regularized logistic regression
    for yb, txb in random_batches(y, tx, batch_size, max_iters):
        # updating the weights
        grad=log_likelihood_gradient(yb, txb, w)+2*lambda_*w
        w-=gamma*grad
        
    loss= log_likelihood_loss(y, tx, w)+lambda_*np.squeeze(w.T@w)
    return w, loss
    
def data_split(y, tx, train_ratio,valid_ratio):
    """
    splits data into training,validation and test sets based on the ratios
    """
    # initializing the seed
    np.random.seed(1)
    
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    # Calculate the indices of the training,validation and test sets
    split = (np.array([train_ratio, valid_ratio]) * len(y)).astype(int).cumsum()
    train_index, valid_index, test_index = np.split(indices,split)
    
    train_x = tx[train_index]
    train_y = y[train_index]
    validation_x = tx[valid_index]
    validation_y = y[valid_index]
    test_x = tx[test_index]
    test_y = y[test_index]
    
    return train_x,train_y,validation_x,validation_y,test_x,test_y
