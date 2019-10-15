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