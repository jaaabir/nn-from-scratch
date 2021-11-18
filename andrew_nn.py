import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

class Network(BaseEstimator):
    def __init__(self, learning_rate = 0.01, epoches = 30, activations = [], layers  = []):
        self.learning_rate = learning_rate
        self.layers = layers
        self.n_layers = len(layers) - 1            
        self.activations = activations
        self.epoches = epoches
        self.params = {}
        self.costs = []
        
    def initialize_parameters_deep(self,layer_dims):
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)

        for l, units in enumerate(zip(layer_dims[:-1], layer_dims[1:])):

            current_units, next_units = units
            # initializer = np.sqrt(2 / current_units) if (l + 1) < (L - 1) else 0.01
            initializer = 0.01
            parameters['W' + str(l + 1)] = np.random.randn(next_units, current_units) * initializer
            parameters['b' + str(l + 1)] = np.zeros((next_units, 1))

        return parameters


    def linear_forward(self,A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            A, activation_cache = relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self,X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2                  
        for l in range(1, L):
            A_prev = A 
            W = parameters[f'W{l}']
            b = parameters[f'b{l}']
            A, cache = self.linear_activation_forward(A_prev, W, b, 'relu')
            caches.append(cache)

        W = parameters[f'W{L}']
        b = parameters[f'b{L}']
        AL, cache = self.linear_activation_forward(A, W, b, 'sigmoid')
        caches.append(cache)

        return AL, caches
    def compute_cost(self, AL, Y):
    
        m = Y.shape[1]
        cost = - np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)     
        return cost

    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.mean(dZ, keepdims = True, axis = 1)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ =  relu_backward(dA, activation_cache)
            dA_prev, dW, db =  self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ =  sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db =  self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches) 
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]

        dA_prev_temp, dW_temp, db_temp =  self.linear_activation_backward(dAL, current_cache, 'sigmoid')
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads[f"dA{l + 1}"], current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp


        return grads

    def update_parameters(self, params, grads, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads[f'dW{l  + 1}'] 
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads[f'db{l  + 1}']  

        return parameters
    
    def GD(self, X, y):
        params = self.initialize_parameters_deep(self.layers)
        for epoch in tqdm(range(self.epoches)):
            al, caches = self.L_model_forward(X, params)
            cost = self.compute_cost(al, y)
            grads = self.L_model_backward(al, y, caches)
            params = self.update_parameters(params, grads, self.learning_rate)
    
            if epoch % 100 == 0:
                self.costs.append(cost)
                
        self.params = params
                
        return 
                
    def fit(self, X, y):
        self.GD(X, y)
        return self
        
    def predict(self, X):
        al, caches = self.L_model_forward(X, self.params)
        return al