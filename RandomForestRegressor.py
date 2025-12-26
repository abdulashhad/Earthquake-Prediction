import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dataset.csv")
data = np.array(data)

X = data[:, 0:-1]
y = data[:, -1]
y = y.astype('float')
X = X.astype('float')

# Convert labels to integers
X = X.round().astype(int)
y = y.round().astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and fit the regressor
rfc = RandomForestRegressor(n_estimators=100)
rfc.fit(X_train, y_train)

# Predict on test data
y_pred_test = rfc.predict(X_test)

# Making Predictions
input_data = np.array([[-10.6812, 161.327, 40]])
prediction = rfc.predict(input_data)

print("Prediction is:", prediction)

if prediction >= 5.5:
    print("Danger")
elif 4 < prediction < 5.5:
    print("Low Risk")
else:
    print("No risk")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dataset.csv")
data = np.array(data)

X = data[:, 0:-1]
y = data[:, -1]
y = y.astype('float')
X = X.astype('float')

# Convert labels to integers
X = X.round().astype(int)
y = y.round().astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and fit the regressor
rfc = RandomForestRegressor(n_estimators=100)
rfc.fit(X_train, y_train)

# Predict on test data
y_pred_test = rfc.predict(X_test)

# Making Predictions
input_data = np.array([[-10.6812, 161.327, 40]])
prediction = rfc.predict(input_data)

print("Prediction is:", prediction)

if prediction >= 5.5:
    print("Danger")
elif 4 < prediction < 5.5:
    print("Low Risk")
else:
    print("No risk")