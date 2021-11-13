# Neural network from scratch 

its not a perfect but a basic deep L layer neural network i learned from Andrew-ng deep learning course, its a combination of his implementation and my own.

### Example

```
nn = Network(
    layers = [5, 10, 10, 1], # 3 layers neural network
    activations = ['relu', 'relu', 'sigmoid']
    learning_rate = 0.005,
    epoches = 3000
)

# layers      :==>  [input_layer_unit_size, 1st_hidden_layer_unit_size, ... , n_hidden_layer_unit_size, output_layer_unit_size] 
# activations :==>  relu, sigmoid

X_train = X_train.T
X_test = X_test.T
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1,-1)

nn.fit(X_train, y_train)
pred = (nn.predict(X_test) > 0.5 ).astype(int)
```
