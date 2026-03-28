import numpy as np
from sklearn.linear_model import LinearRegression

# ১. Data Setup (X = input, y = output)
# Dhore nei, X holo barir room er shonkha, ar y holo barir dam (lakh taka-y)
# Example: 1 room = 10 lakh, 2 rooms = 20 lakh...
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

# ২. Model Create kora
model = LinearRegression()

# ৩. AI-ke Train kora (Fit kora)
model.fit(X, y)

# ৪. Prediction kora
# Akhon amra AI-ke jiggesh korbo: "Jodi 7-ta room thake, tobe dam koto hobe?"
rooms_to_predict = np.array([[7]])
prediction = model.predict(rooms_to_predict)

print(f"Prediction for 7 rooms: {prediction[0]} Lakh Taka")

# Coefficient check (AI ki logic shikhlol)
print(f"AI er logic (Weight): {model.coef_[0]}")