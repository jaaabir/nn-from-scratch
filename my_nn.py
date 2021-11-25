import numpy as np
import pandas as pd 
from numpy import random
from numpy.random.mtrand import beta, random_sample 
from sklearn.base import BaseEstimator
from tqdm import tqdm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(da, z):
    return da * sigmoid(z) * (1 - sigmoid(z))
    
def relu(z):
    return np.maximum(0, z)

def relu_prime(da, z):
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    return dz

class Network(BaseEstimator):
    def __init__(self, learning_rate = 0.01, epoches = 30, activations = [], layers  = [], 
                optimizer = 'adam', batch_size = 64, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, decay_rate = 0.5, 
                random_state = None, regularization = 'l2', keep_prob = 1, lambd = 0, t = 2, y_reshape = False):
        
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

        self.learning_rate = learning_rate
        self.layers = layers
        self.n_layers = len(layers) - 1            
        self.activations = activations
        self.epoches = epoches
        self.weights, self.biases = self.initialize_parameters()
        self.cache = self.initialize_cache()
        self.v, self.s = self.initialize_ewa()
        self.costs = []
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_rate = decay_rate
        self.regularization = regularization
        self.keep_prob = keep_prob
        self.epsilon = epsilon
        self.t = t
        self.y_reshape = y_reshape
        
    def initialize_parameters(self):
        w = [
            np.random.randn(next_layer, current_layer) * 0.01 for current_layer, next_layer in zip(self.layers[:-1], self.layers[1:]) 
        ]
        b = [
            np.zeros((next_layer, 1)) for next_layer in self.layers[1:]
        ]
        
        return w, b

    def initialize_ewa(self):
        v = {}
        s = {}
        for i in range(len(self.weights)):
            v[f'dw{i + 1}'] = np.zeros(self.weights[i].shape)
            v[f'db{i + 1}'] = np.zeros(self.biases[i].shape)
            s[f'dw{i + 1}'] = np.zeros(self.weights[i].shape)
            s[f'db{i + 1}'] = np.zeros(self.biases[i].shape)

        return v, s
    
    def initialize_cache(self):
        c = {}
        for i in range(len(self.weights)):
            c[f'z{i + 1}']   = None
            c[f'activation{i + 1}'] = None
            c[f'w{i + 1}']   = self.weights[i]
            c[f'b{i + 1}']   = self.biases[i]
        
            c[f'da{i + 1}']  = None 
            c[f'dw{i + 1}']  = None
            c[f'db{i + 1}']  = None
            
        return c
    
    def activate(self, z, activation):
        if activation == 'sigmoid':
            return sigmoid(z)
        elif activation == 'relu':
            return relu(z)
    
    def forward(self, x):
        a_prev = x
        self.cache[f'activation0'] = a_prev
        # for i, params in enumerate(zip(self.weights, self.biases)):
        for i in range(self.n_layers):
            # w, b = params
            w, b = self.cache[f"w{i + 1}"], self.cache[f"b{i + 1}"]
            z = np.dot(w, a_prev) + b
            a_prev = self.activate(z, self.activations[i])
            self.cache[f'z{i + 1}'] = z
            self.cache[f'activation{i + 1}'] = a_prev
        
        return a_prev
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.mean(dZ, keepdims = True, axis = 1)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def activate_prime(self, da, layer, activation):
        if activation == 'sigmoid':
            dz = sigmoid_prime(da, self.cache[f"z{layer}"])
        elif activation == 'relu':
            dz = relu_prime(da, self.cache[f"z{layer}"])
            
        cache = self.cache[f'activation{layer - 1}'], self.cache[f'w{layer}'], self.cache[f'b{layer}']
        return self.linear_backward(dz, cache)
    
    def backward(self, a, y):
        
        L = self.n_layers
        if self.y_reshape:
            y = y.reshape(a.shape)
        da = - (np.divide(y, a) - np.divide((1 - y), (1 - a)))
        self.cache["da" + str(L)] = da
        
        da_prev, dw, db = self.activate_prime(da, L, self.activations[ L - 1 ])
        self.cache["da" + str(L-1)] = da_prev
        self.cache["dw" + str(L)] = dw
        self.cache["db" + str(L)] = db
        
        for l in range(L - 1, 0, -1):
            da = self.cache[f'da{l}']
            da_prev, dw, db = self.activate_prime(da, l, self.activations[ l - 1 ])
            self.cache["da" + str(l-1)] = da_prev
            self.cache["dw" + str(l)] = dw
            self.cache["db" + str(l)] = db
            
        return
        
    def compute_cost(self, a, y):
        if self.y_reshape:
            y = y.reshape(a.shape)
        loss = np.multiply(y, np.log(a)) + np.multiply((1 - y), np.log(1 - a))
        return -np.mean(loss)

    def update_parameters(self):
        for l in range(1, self.n_layers + 1):
            self.cache[f"w{l}"] = self.cache[f"w{l}"] - (self.learning_rate * self.cache[f"dw{l}"])
            self.cache[f"b{l}"] = self.cache[f"b{l}"] - (self.learning_rate * self.cache[f"db{l}"]) 

    def update_paramters_with_adam(self):
        v_corrected = {}
        s_corrected = {}

        for l in range(1, self.n_layers + 1):
            self.v[f'dw{l}'] = (self.beta1 * self.v[f'dw{l}']) + ((1 - self.beta1) * self.cache[f'dw{l}'])
            self.v[f'db{l}'] = (self.beta1 * self.v[f'db{l}']) + ((1 - self.beta1) * self.cache[f'db{l}'])

            v_corrected[f'dw{l}'] = self.v[f'dw{l}'] / ( 1 - self.beta1**self.t )
            v_corrected[f'db{l}'] = self.v[f'db{l}'] / ( 1 - self.beta1**self.t )

            self.s[f'dw{l}'] = (self.beta2 * self.s[f'dw{l}']) + ((1 - self.beta2) * self.cache[f'dw{l}']**2)
            self.s[f'db{l}'] = (self.beta2 * self.s[f'db{l}']) + ((1 - self.beta2) * self.cache[f'db{l}']**2)

            s_corrected[f'dw{l}'] = self.s[f'dw{l}'] / ( 1 - self.beta2**self.t )
            s_corrected[f'db{l}'] = self.s[f'db{l}'] / ( 1 - self.beta2**self.t )

            self.cache[f"w{l}"] = self.cache[f"w{l}"] - (self.learning_rate * (v_corrected[f'dw{l}'] / (np.sqrt(s_corrected[f'dw{l}']) + self.epsilon)))
            self.cache[f"b{l}"] = self.cache[f"b{l}"] - (self.learning_rate * (v_corrected[f'db{l}'] / (np.sqrt(s_corrected[f'db{l}']) + self.epsilon)))

    def GD(self, X, y):
        for epoch in tqdm(range(self.epoches)):
            yhat = self.forward(X)
            cost = self.compute_cost(yhat, y)
            self.backward(yhat, y)
            
            self.update_parameters()
                
            if epoch % 100 == 0:
                self.costs.append(cost)
                
        return 

    def adam(self, X, Y):
        for epoch in tqdm(range(self.epoches)):
            idx = np.random.permutation(X.shape[1])
            shuffled_X = X[:, idx]
            shuffled_y = Y[:, idx]

            st = 0
            ed = self.batch_size
            iter_per_batch_size = (X.shape[1] // self.batch_size) + 1

            for _ in range(iter_per_batch_size):
                batch_X = shuffled_X[:, st : ed]
                batch_Y = shuffled_y[:, st : ed]

                if batch_X.shape[1] > 0:
                    yhat = self.forward(batch_X)
                    cost = self.compute_cost(yhat, batch_Y)
                    self.backward(yhat, batch_Y)

                    self.update_paramters_with_adam()

                    st  = ed
                    ed += self.batch_size 

                    if epoch % 100 == 0:
                        self.costs.append(cost)


                
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.optimizer == 'gd':
            self.GD(X, y)
        else:
            self.adam(X, y)

        return self
        
    def predict(self, X):
        return self.forward(X)