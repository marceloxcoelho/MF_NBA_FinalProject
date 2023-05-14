import pandas as pd  # Importing the pandas library for data manipulation
import numpy as np  # Importing the numpy library for numerical computations
import sklearn  # Importing the scikit-learn library for machine learning tasks
import seaborn as sns  # Importing the seaborn library for visualization
from sklearn.model_selection import train_test_split  # Importing the train_test_split function from scikit-learn for splitting the data
from sklearn.preprocessing import StandardScaler  # Importing the StandardScaler class from scikit-learn for feature scaling
from sklearn.metrics import accuracy_score, log_loss  # Importing accuracy_score and log_loss functions from scikit-learn for model evaluation
from sklearn.neural_network import MLPClassifier  # Importing the MLPClassifier class from scikit-learn for training a neural network model
from sklearn.svm import LinearSVC, SVC  # Importing the LinearSVC and SVC classes from scikit-learn for training SVM models
from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier class from scikit-learn for training a random forest model
import time  # Importing the time module for tracking execution time

pd.set_option('display.max_columns', 500)  # Setting pandas options to display a maximum of 500 columns
pd.set_option('display.max_rows', 500)  # Setting pandas options to display a maximum of 500 rows


df = pd.read_csv('../Data/NBAGameDataset.csv')  # Reading the data from the CSV file and storing it in a pandas DataFrame

total = df.isnull().sum().sort_values(ascending=False)  # Calculating the total number of missing values in each column and sorting them in descending order
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)  # Calculating the percentage of missing values in each column and sorting them in descending order
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # Combining the total and percentage of missing values into a single DataFrame
print("Missing Values:")  # Printing the string "Missing Values"
print(missing_data.head())  # Printing the first 5 rows of the DataFrame "missing_data"


player_cols = []  # Creating an empty list to store column names containing 'Player'
for c in df.columns:  # Iterating over each column in the DataFrame
    temp = c  # Storing the column name in a temporary variable
    c = c.split(' ')  # Splitting the column name by space
    if len(c) > 1 and c[1] == 'Player':  # Checking if the column name contains 'Player' as the second part after splitting
        player_cols.append(temp)  # Appending the column name to the list "player_cols"

df = df.drop(columns=player_cols)  # Dropping the columns specified in the "player_cols" list from the DataFrame


X = df.drop(columns=[  # Creating the feature matrix X by dropping specified columns from the DataFrame
    'Match Up', 'Game Date', 'W/L',
    'H Win Avg', 'V Win Avg', 'H Win Avg Last 8', 'V Win Avg Last 8', 'H PF', 'V PF', 'H TOV', 'V TOV',
    'V OREB Last 8', 'H OREB Last 8', 'H FT% Last 8', 'V FT% Last 8'
])

y = df['W/L']  # Creating the target vector y by selecting the 'W/L' column from the



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# Splitting the data into training and testing sets using train_test_split function from scikit-learn.
# X_train and X_test will contain the features, y_train and y_test will contain the corresponding labels.
# test_size=0.2 specifies that 20% of the data will be used for testing, while 80% will be used for training.
# random_state=42 sets a specific random seed for reproducibility, and shuffle=True shuffles the data before splitting.

scalar = StandardScaler()
# Creating an instance of the StandardScaler class from scikit-learn for feature scaling.

X_train = scalar.fit_transform(X_train)
# Fitting the StandardScaler on the training data and transforming it to standardized values.
# This ensures that each feature has zero mean and unit variance in the training set.

X_test = scalar.transform(X_test)
# Transforming the testing data using the previously fitted StandardScaler.
# This applies the same scaling transformation as done on the training set.

# Neural Network
start_timeNN = time.time()  # Record the current time
# Getting the current timestamp as the starting point of execution time measurement for the Neural Network model.

classifier_NN = MLPClassifier((8, 12, 10), max_iter=5000, solver='sgd', alpha=0.1, activation='relu', learning_rate='adaptive', random_state=42)
# Creating an instance of the MLPClassifier class from scikit-learn for training a neural network model.
# This specifies a neural network architecture with (8, 12, 10) hidden layers, maximum of 5000 iterations,
# stochastic gradient descent (sgd) solver, 0.1 alpha (regularization parameter), 'relu' activation function,
# 'adaptive' learning rate, and a specific random seed for reproducibility.

classifier_NN.fit(X_train, y_train)
# Fitting the neural network model to the training data.

y_pred_NN = classifier_NN.predict(X_test)
# Predicting the labels for the testing data using the trained neural network model.

print("Accuracy on Neural Network: {0}".format(accuracy_score(y_test, y_pred_NN)))
# Calculating the accuracy of the neural network model by comparing the predicted labels with the true labels of the testing data.

end_timeNN = time.time()  # Record the current time again
# Getting the current timestamp as the ending point of execution time measurement for the Neural Network model.

execution_timeNN = end_timeNN - start_timeNN  # Calculate the difference to get the execution time
# Calculating the execution time for the Neural Network model by taking the difference between the starting and ending timestamps.

print("Execution time for NN: ", execution_timeNN, " seconds")
# Printing the execution time for the Neural Network model.

#Random Forest
start_timeRF = time.time()  # Record the current time

# Create a Random Forest Classifier object with 400 estimators, maximum depth of 12, utilizing all available processors, and a random state of 69
classifier_RF = RandomForestClassifier(n_estimators=400, max_depth=12, n_jobs=-1, random_state=69)

# Fit the Random Forest Classifier using the training data
classifier_RF.fit(X_train, y_train)

# Use the trained Random Forest Classifier to make predictions on the test data
y_pred_RF = classifier_RF.predict(X_test)

# Print the accuracy of the Random Forest Classifier by comparing the predicted labels with the true labels
print("Accuracy on Random Forest Classifier: {0}".format(accuracy_score(y_test, y_pred_RF)))

end_timeRF = time.time()  # Record the current time again

# Calculate the difference between the end time and the start time to get the execution time
execution_timeRF = end_timeRF - start_timeRF

# Print the execution time for the Random Forest Classifier
print("Execution time for RF: ", execution_timeRF, " seconds")

