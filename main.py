from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the dataset
iris = load_iris()

# Take a look at our data
print(iris.data)

# Show attribute names
print(iris.feature_names)

# Show target vector: Each row corresponds to what class
print(iris.target)

# Print Target vector names: such as 0 is Setosa
print(iris.target_names)

# Print Dimensions of the data
print(iris.data.shape)

x = iris.data
y = iris.target
print(x.shape, y.shape)

# K Nearest Neighbor classifier with K = 3
knn = KNeighborsClassifier(n_neighbors=5)
print(knn)
# Fit the model with training data and training labels
knn.fit(x, y)

# Fit model with new input data
xNew = [[3, 5, 4, 2], [5, 4, 3, 2]]
# Print the output of the new prediction
print(knn.predict(xNew))

# Evaluate the accuracy of predicting
print(metrics.accuracy_score(y, knn.predict(x)))