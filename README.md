# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries.
2. Load the California Housing dataset and split it into training and testing sets.
3. Scale the input features using StandardScaler.
4. Train the SGD Regressor model using the training data and predict the house prices.
5. Display the predicted values and evaluate the model performance.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MAALINI B N
RegisterNumber:  212224060136
*/
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SGDRegressor(max_iter=1000, learning_rate='constant',
                     eta0=0.01, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nSample Predictions:")
for i in range(5):
    print("Actual:", round(y_test[i], 2),
          "Predicted:", round(y_pred[i], 2))
```

## Output:

<img width="420" height="205" alt="image" src="https://github.com/user-attachments/assets/60385f4c-9c78-478c-a60a-7528eff499b5" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
