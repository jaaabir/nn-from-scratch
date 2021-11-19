import numpy as np 
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
    def __init__(self, learning_rate = 0.01, epoches = 30, activations = [], layers  = []):
        self.learning_rate = learning_rate
        self.layers = layers
        self.n_layers = len(layers) - 1            
        self.activations = activations
        self.epoches = epoches
        self.weights, self.biases = self.initialize_parameters()
        self.cache = self.initialize_cache()
        self.costs = []
        
    def initialize_parameters(self):
        np.random.seed(3)
        w = [
            np.random.randn(next_layer, current_layer) * 0.01 for current_layer, next_layer in zip(self.layers[:-1], self.layers[1:]) 
        ]
        b = [
            # np.random.randn(next_layer, 1) for next_layer in self.layers[1:]
            np.zeros((next_layer, 1)) for next_layer in self.layers[1:]
        ]
        
        return w, b
    
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
        y = y.values.reshape(a.shape)
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
        y = y.values.reshape(a.shape)
        loss = np.multiply(y, np.log(a)) + np.multiply((1 - y), np.log(1 - a))
        return -np.mean(loss)
    
    def GD(self, X, y):
        for epoch in tqdm(range(self.epoches)):
            yhat = self.forward(X)
            cost = self.compute_cost(yhat, y)
            self.backward(yhat, y)
            
            for l in range(1, self.n_layers + 1):
                self.cache[f"w{l}"] = self.cache[f"w{l}"] - (self.learning_rate * self.cache[f"dw{l}"])
                self.cache[f"b{l}"] = self.cache[f"b{l}"] - (self.learning_rate * self.cache[f"db{l}"]) 
                
            if epoch % 100 == 0:
                self.costs.append(cost)
                
        return 
                
    def fit(self, X, y):
        self.GD(X, y)
        return self
        
    def predict(self, X):
        return self.forward(X)