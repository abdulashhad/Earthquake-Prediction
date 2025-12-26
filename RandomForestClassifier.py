import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("earthquake.csv")

data = np.array(data)
print(data)
X = data[:, 0:-1]
y = data[:, -1]
y = y.astype('float')
X = X.astype('float')

# Convert labels to integers
y = y.round().astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and fit the classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Accuracy on test data:
y_pred_test = rfc.predict(X_test)
test_data_accuracy = accuracy_score(y_pred_test, y_test)
print("Accuracy on Test Data: ", test_data_accuracy)

# Making Predictive System
input_data = (39.13,43.19,42)
# Conversion of input data to numpy array for faster processing
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance and label for this one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = rfc.predict(input_data_reshaped)
print("Prediction is:", prediction)

if prediction >= 5.5:
    print("Danger")
elif 4 < prediction < 5.5:
    print("Low Risk")
else:
    print("No risk")
