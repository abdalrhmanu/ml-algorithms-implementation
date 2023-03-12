import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from knn import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

classifier = KNN(k=5)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)

print(f"Accuracy: {acc:.2f}%")



# print(X_train.shape)
# print(X_train[0])

# print(y_train.shape)
# print(y_train)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=20)
# plt.show()