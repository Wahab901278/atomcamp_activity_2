import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('salary_data.csv')
X = data['YearsExperience'].values.reshape(-1, 1)
Y = data['Salary'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

mse = mean_squared_error(y_test, y_predict)

print(f"Mean Square Error: {mse}")