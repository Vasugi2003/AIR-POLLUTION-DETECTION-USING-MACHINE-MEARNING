

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
dataset = pd.read_csv('/content/station_day.csv')

import matplotlib
import matplotlib .pyplot as plt
import seaborn as sns


dataset.head()

sns.catplot(x = "AQI_Bucket", kind= "count", palette = "ch: 2.87", height=5, aspect=1.1, data = dataset)

dataset = dataset.dropna(subset=["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene","AQI","AQI_Bucket"])

X = dataset.drop(['AQI_Bucket','StationId','Date'], axis=1)

dataset.AQI_Bucket

dataset['AQI_Bucket'].value_counts()

from sklearn.preprocessing import LabelEncoder
class_labels = dataset.AQI_Bucket
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)
continuous_labels = label_encoder.transform(class_labels)
print(class_labels.unique())
print(continuous_labels)
print(continuous_labels)

y = continuous_labels

print(X)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

from sklearn.preprocessing import StandardScaler
xt = StandardScaler().fit_transform(X)

clf = SVC(kernel='linear')
clf.kernel

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:',accuracy)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MEAN SQUARED ERROR =",mse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MEAN ABSOLUTE ERROR =",mae)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='weighted')
print("PRECISION =",precision)

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred,average='weighted')
print("RECALL VALUE = ",recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred,average='weighted')
print("F1 SCORE =",f1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

import pickle

with open('clf.pkl', 'wb') as file:
    pickle.dump(clf, file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

# Load the model from the file
with open('clf.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model to make predictions
predictions = loaded_model.predict(X_test)

# Print the predictions
print(predictions)

# Load the model and the label encoder from the files
with open('clf.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    loaded_label_encoder = pickle.load(encoder_file)

# Use the loaded model to make predictions
predictions = loaded_model.predict(X_test)

# Convert the continuous labels back to original class labels
original_labels = loaded_label_encoder.inverse_transform(predictions)

# Print the predictions
print(original_labels)

