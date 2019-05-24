import numpy as np

def initialize_parameters(n_x):
    """
    get n_x as the number of features and initialize W and b randomly and output dictionary that contains W and b
    """
    W = np.random.randn(n_x, 1) * 0.001
    b = 0

    paramters = {'W': W,
                 'b': b}

    return paramters

def forward_propagation(X, parameters):
    """
    Z = W.T * X + b
    """

    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W.T, X) + b

    m = X.shape[1]

    assert(Z.shape[1] == m)

    return Z

def cost(Z, Y):
    """
    compute the square loss
    """
    m = Z.shape[1]

    cost = 1/(2*m) * np.sum(np.square(Z - Y))

    return cost

def back_propagation(X, Y, Z):

    n_x = X.shape[0]
    m = X.shape[1]

    dZ = 1/m * (Z - Y)

    dW = np.dot(X, dZ.T)
    db = np.sum(dZ)

    assert (dW.shape[0] == n_x)

    cache = {'dW' : dW,
             'db' : db}

    return cache

def model(X, Y, learning_rate= 0.009, epoch=5):
    """
    X : input with shape of (n_x, m) that n_x is the number of features and m is the number of training examples
    Y : output with shape of (1, m) that m is the number of training examples
    """

    n_x = X.shape[0]
    m = X.shape[1]

    parameters = initialize_parameters(n_x)

    Z = np.zeros((1,m))
    W = parameters['W']
    b = parameters['b']
    costs = []

    for i in range(epoch):
        
        parameters['W'] = W
        parameters['b'] = b
        
        Z = forward_propagation(X, parameters)

        costs.append(cost(Z, Y))

        cache = back_propagation(X, Y, Z)

        dW = cache['dW']
        db = cache['db']

        W -= learning_rate * dW
        b -= learning_rate * db


    return Z, costs, parameters


