from sklearn.linear_model import LinearRegression
import numpy as np

# Возраст и соответствующий рост
age = np.array([5, 8, 12, 18, 25, 30]).reshape(-1, 1)
height = np.array([110, 130, 150, 170, 178, 180])

model = LinearRegression()
model.fit(age, height)

# Предсказание для возраста 20 лет
prediction = model.predict([[20]])
print("Предсказанный рост:", prediction[0])
