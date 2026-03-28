import numpy as np
from sklearn.linear_model import LinearRegression


# 1 room = 10 lakh, 2 rooms = 20 lakh.
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

model = LinearRegression()

model.fit(X, y)

rooms_to_predict = np.array([[7]])
prediction = model.predict(rooms_to_predict)

print(f"Prediction for 7 rooms: {prediction[0]} Lakh Taka")

print(f"AI er logic (Weight): {model.coef_[0]}")