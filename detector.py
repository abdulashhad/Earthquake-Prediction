import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# # import pickle
# from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("earthquake.csv")

# data = np.array(data)
# # print(data)
# X = data[:, 0:-1]
# y = data[:, -1]
X=data.drop(['xm'],axis=1)
y=data['xm']
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print(X.shape,X_train.shape, X_test.shape)

#Model Training->Logistic Regression Model
model=LogisticRegression()
#training the logistic regression model with training data
model.fit(X_train,y_train)
# print(X_train)
# print(y_train)

#model Evalution
#Accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=100*accuracy_score(X_train_prediction,y_train)
print("Accuracy on Training Data: ",training_data_accuracy)

#Accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=100*accuracy_score(X_test_prediction,y_test)
print("Accuracy on Test Data: ",test_data_accuracy)

#Making Predictive System
input_data=(37.09,44.87,5.4)
#Conversion of i/p data to numpy array for faster processing
input_data_as_numpy_array=np.asarray(input_data)

#reshape the np array as we are predicting for one instance 1 instance and label for this one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print("Prediction is :",prediction)

if(prediction>= 5.5):
    print("Danger")
elif(prediction>4 & prediction<5.5):
    print("Low Risk")
else:
    print("No risk")

# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)

# print(metrics.accuracy_score(y_test, y_pred))


# pickle.dump(rfc,open('model.pkl','wb'))



 