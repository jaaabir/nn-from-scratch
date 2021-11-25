import pandas as pd 
import numpy as np
from pandas.core.algorithms import mode
from scipy.sparse.construct import rand 
from my_nn import Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 

df = pd.read_csv('heart.csv')
x = df.drop(['target'], axis = 1)
y = df.target
xt, xtest, yt, ytest = train_test_split(x, y, test_size = 0.2, stratify=y, random_state = 123)

# comparision between GD vs adam optimizer ( adam is so much better )
model = Network(
    layers = [13, 10, 1],
    activations = ['relu', 'sigmoid'],
    epoches = 5000,
    learning_rate = 0.01,
    optimizer = 'adam',
    random_state= 3
)


# model = Network(
#     layers = [13, 10, 1],
#     activations = ['relu', 'sigmoid'],
#     epoches = 5000,
#     learning_rate = 0.01,
#     random_state = 3,
#     optimizer = 'gd'
# )


yt = yt.values.reshape(1, -1)   
model.fit(xt.T, yt)
pred = np.squeeze((model.predict(xtest.T) > 0.5).astype(int))
plt.plot(model.costs)
plt.show()
print(classification_report(ytest, pred))