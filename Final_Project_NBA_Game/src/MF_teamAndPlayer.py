import pandas as pd  # Importing the pandas library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Importing the train_test_split function from scikit-learn for splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Importing the StandardScaler class from scikit-learn for feature scaling
from sklearn.metrics import accuracy_score, log_loss  # Importing accuracy_score and log_loss functions from scikit-learn for model evaluation
from sklearn.neural_network import MLPClassifier  # Importing the MLPClassifier class from scikit-learn for building a neural network model
from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier class from scikit-learn for building a random forest model
import time  # Importing the time module for recording time

pd.set_option('display.max_columns', 500)  # Setting pandas option to display a maximum of 500 columns
pd.set_option('display.max_rows', 500)  # Setting pandas option to display a maximum of 500 rows

df = pd.read_csv('../data/NBAGameDataset.csv')  # Reading the dataset from a CSV file and storing it in a pandas DataFrame

total = df.isnull().sum().sort_values(ascending=False)  # Counting the total number of missing values in each column and sorting them in descending order
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)  # Calculating the percentage of missing values in each column and sorting them in descending order
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # Concatenating the total and percent columns into a single DataFrame
print("Missing Values:")
print(missing_data.head())  # Printing the DataFrame showing the columns with the most missing values


X = df.drop(columns=[  # Creating a new DataFrame X by dropping the specified columns from the original DataFrame
    'Match Up', 'Game Date', 'W/L',
    'H Player FG% Last 8', 'H Player FG%', 'V Player FG% Last 8', 'V Player FG%',  
    'H Player FG% Last 8', 'H Player FG%', 'V Player FG% Last 8', 'V Player FG%',
    'H Player 3P% Last 8', 'H Player 3P%', 'V Player 3P% Last 8', 'V Player 3P%',
    'H Player FT% Last 8', 'H Player FT%', 'V Player FT% Last 8', 'V Player FT%',
    'H OREB Last 8', 'H OREB', 'V OREB Last 8', 'V OREB',
    'H DREB Last 8', 'H DREB', 'V DREB Last 8', 'V DREB',
    'H Player BLK Last 8', 'H Player BLK', 'V Player BLK Last 8', 'V Player BLK',
    'H AST Last 8', 'H Player AST', 'V AST Last 8', 'V Player AST',
    'H STL Last 8', 'H Player STL', 'V STL Last 8', 'V Player STL',
    'H Player TOV Last 8', 'H Player TOV', 'V Player TOV Last 8', 'V Player TOV',
    'H PF Last 8', 'H PF', 'V PF Last 8', 'V PF',
    'H Player Pts Diff Avg', 'V Player Pts Diff Avg',
    'H Player Pts Diff Avg Last 8', 'V Player Pts Diff Avg Last 8',
    'H Player +/- Last 8', 'V Player +/- Last 8',
    'H Player +/-', 'V Player +/-',
    'H Player AST Last 8', 'V Player AST Last 8', 'H Player PF', 'V Player PF',
    'H Player DREB','V Player DREB',    
    'H TOV', 'V TOV',
    'H Player STL Last 8', 'V Player STL Last 8', 
    'H 3P%', 'V 3P%',
    'H Player PF', 'V Player PF',
])

y = df['W/L']  # Creating a new Series y by extracting the 'W/L' column from the original DataFrame

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # Splitting the data into training and testing sets using train_test_split function from scikit-learn

scalar = StandardScaler()  # Creating an instance of StandardScaler for feature scaling
X_train = scalar.fit_transform(X_train)  # Scaling the training features using fit_transform method of StandardScaler
X_test = scalar.transform(X_test)  # Scaling the testing features using transform method of StandardScaler


# Neural Network
start_timeNN = time.time()  # Record the current time
# Creating an instance of MLPClassifier for neural network classification with specified parameters
classifier_NN = MLPClassifier((8, 12, 10), max_iter=5000, solver='sgd', alpha=0.1, activation='relu', learning_rate='adaptive', random_state=42)
classifier_NN.fit(X_train, y_train)  # Training the neural network model using fit method
y_pred_NN = classifier_NN.predict(X_test)  # Predicting the labels for the test set using the trained model
# Calculating and printing the accuracy of the neural network model
print("Accuracy on Neural Network: {0}".format(accuracy_score(y_test, y_pred_NN)))
end_timeNN = time.time()  # Record the current time again
execution_timeNN = end_timeNN - start_timeNN  # Calculate the difference to get the execution time
# Printing the execution time for the neural network model
print("Execution time for NN: ", execution_timeNN, " seconds")


# Random Forest
start_timeRF = time.time()  # Record the current time
# Creating an instance of RandomForestClassifier for random forest classification with specified parameters
classifier_RF = RandomForestClassifier(n_estimators=400, max_depth=6, n_jobs=-1, random_state=42)
classifier_RF.fit(X_train, y_train)  # Training the random forest model using fit method
y_pred_RF = classifier_RF.predict(X_test)  # Predicting the labels for the test set using the trained model
# Calculating and printing the accuracy of the random forest model
print("Accuracy on Random Forest Classifier: {0}".format(accuracy_score(y_test, y_pred_RF)))
end_timeRF = time.time()  # Record the current time again
execution_timeRF = end_timeRF - start_timeRF  # Calculate the difference to get the execution time
# Printing the execution time for the random forest model
print("Execution time for RF: ", execution_timeRF, " seconds")

