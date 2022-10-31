# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import datetime

# useful tool functions
def compute_loss(y, tx, w):
    """Calculate the loss using mse."""
    e = y - tx.dot(w)
    return calculate_mse(e)


def calculate_mse(err):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(err ** 2)


def compute_gradient_mse(y, tx, w):
    """Compute the gradient of mse."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


"""Function implementation"""

# 1
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # Parameters to store weights and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient_mse(y, tx, w)
        # w update by gradient descent
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)
    print(
        "Gradient Descent: loss={l}, w0={w0}, w1={w1}".format(l=loss, w0=w[0], w1=w[1])
    )
    return loss, w


# 2
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    # Define batch_size
    batch_size = 1
    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            # compute a stochastic gradient and loss
            grad, err = compute_gradient_mse(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)

    print("SGD: loss={l}, w0={w0}, w1={w1}".format(l=loss, w0=w[0], w1=w[1]))
    return loss, w


# 3
def least_squares(y, tx):
    """Least squares regression using normal equations"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse


# 4
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse


# tool function for logistic regression descent
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def c_loss(y, tx, w):
    """Logistic loss"""
    predict = sigmoid(tx.dot(w))
    loss = -(y.T.dot(np.log(predict)) + (1.0 - y).T.dot(np.log(1.0 - predict))) / len(
        tx
    )
    return loss


def c_gradient(y, tx, w):
    """gradient of logistic regression"""
    predict = sigmoid(tx.dot(w))
    gradient = (tx.T.dot(predict - y)) / len(tx)
    return gradient


# 5
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y ∈ {0, 1}])"""
    # Define parameters to store w and loss
    losses = []
    w = initial_w
    threshold = np.exp(-32)
    loss = 0.0

    for n_iter in range(max_iters):
        grad = c_gradient(y, tx, w)  # previously calculate_gradient
        w = w - (gamma * grad)
        loss = c_loss(y, tx, w)  # previously calculate_loss
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss, w


# 6
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ∥w∥2)"""
    # Define parameters to store w and loss
    losses = []
    w = initial_w
    threshold = np.exp(-32)
    loss = 0

    for n_iter in range(max_iters):
        gradient = c_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - (gamma * gradient)
        loss = c_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss, w


#additional functions
def c_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
    """
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step of Newton's method.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar

    """
    gradient = c_gradient(y, tx, w)
    hessian = c_hessian(y ,tx ,w)
    w = w - (gamma* np.linalg.solve(hessian, gradient))
    loss = c_loss(y, tx, w)
    return loss, w

#7
def logistic_regression_newton_method(y, tx, lambda_, initial_w, max_iter, gamma =1):
    # init parameters
 
    threshold = 1e-8
    losses = []
    loss = 0
    w = initial_w
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        #if iter % 1 == 0:
            #print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold: 
            break
    return loss, w
