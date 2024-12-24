# Importing necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import models, layers

# Load the dataset (replace with the correct dataset path)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/health-insurance.csv"
data = pd.read_csv(url, header=None)
data.columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'expenses']

# Convert categorical data (sex, smoker, region) to numbers
label_encoder = LabelEncoder()

data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])
data['region'] = label_encoder.fit_transform(data['region'])

# Split the data into train and test datasets
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)

# Separate the labels (expenses) from the features
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# Normalize the features 
train_dataset = (train_dataset - train_dataset.mean()) / train_dataset.std()
test_dataset = (test_dataset - test_dataset.mean()) / test_dataset.std()

# Build the Linear Regression model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(train_dataset.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(train_dataset, train_labels, epochs=100, batch_size=32)

# Evaluate the model
loss = model.evaluate(test_dataset, test_labels)

# Print the loss (Mean Absolute Error)
print("Mean Absolute Error on test data: ", loss)

# Make predictions on the test data
predictions = model.predict(test_dataset)

# Compare predictions with actual labels
mae = mean_absolute_error(test_labels, predictions)
print("Final Mean Absolute Error: ", mae)

# Plot the results (optional)
import matplotlib.pyplot as plt
plt.scatter(test_labels, predictions)
plt.xlabel('Actual Expenses')
plt.ylabel('Predicted Expenses')
plt.title('Actual vs Predicted Health Expenses')
plt.show()
