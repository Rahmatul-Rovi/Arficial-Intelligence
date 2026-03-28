from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Data Load
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Setup
model = KNeighborsClassifier(n_neighbors=3)

# 4. Training
model.fit(X_train, y_train)

# 5. Accuracy Check
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Prediction
sample_flower = [[5.1, 3.5, 1.4, 0.2]] 
prediction = model.predict(sample_flower)
print(f"Predicted Flower Species: {iris.target_names[prediction][0]}")