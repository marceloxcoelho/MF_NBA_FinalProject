import pandas as pd  # Importing the pandas library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Importing the train_test_split function from sklearn for splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Importing the StandardScaler class from sklearn for feature scaling
from sklearn.metrics import accuracy_score, log_loss  # Importing accuracy_score and log_loss metrics from sklearn for model evaluation
from sklearn.neural_network import MLPClassifier  # Importing the MLPClassifier class from sklearn for building a Neural Network model
from sklearn.svm import LinearSVC  # Importing the LinearSVC class from sklearn for building a Support Vector Machine model
from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier class from sklearn for building a Random Forest model
import time # To track time during the analysis


pd.set_option('display.max_columns', 500)  # Setting pandas options to display maximum columns up to 500
pd.set_option('display.max_rows', 500)  # Setting pandas options to display maximum rows up to 500

df = pd.read_csv('../Data/NBAGameDataset.csv')  # Reading the data from the CSV file and storing it in a pandas DataFrame

total = df.isnull().sum().sort_values(ascending=False)  # Calculating the total number of missing values in each column and sorting them in descending order
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)  # Calculating the percentage of missing values in each column and sorting them in descending order
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # Combining the total and percentage of missing values into a single DataFrame
print("Missing Values:")  # Printing the string "Missing Values"
print(missing_data.head())  # Printing the first 5 rows of the DataFrame "missing_data"

player_cols = ['W/L', 'H Win Avg', 'V Win Avg', 'H Win Avg Last 8', 'V Win Avg Last 8']  # Creating a list of column names to be selected

for c in df.columns:  # Iterating over each column in the DataFrame
    temp = c  # Storing the column name in a temporary variable
    c = c.split(' ')  # Splitting the column name by space
    if len(c) > 1 and c[1] == 'Player':  # Checking if the column name contains 'Player' as the second part after splitting
        player_cols.append(temp)  # Appending the column name to the list "player_cols"

df = df[player_cols]  # Selecting only the columns specified in the "player_cols" list from the DataFrame and updating it

X = df.drop(columns=['W/L', 'H Player AST Last 8', 'V Player AST Last 8', 'H Player PF', 'V Player PF', 'H Player TOV', 'V Player TOV', 'H Player STL Last 8', 'V Player STL Last 8', 'H Player 3P%', 'V Player 3P%', 'H Player +/- Last 8', 'V Player +/- Last 8', 'H Player +/-', 'V Player +/-'])  # Creating the feature matrix X by dropping the specified columns from the DataFrame

y = df['W/L']  # Creating the target vector y by selecting the 'W/L' column from the DataFrame

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # Splitting the data into training and testing sets using train

scalar = StandardScaler() #create an instance of StandardScaler, which is used to standardize features by removing the mean and scaling to unit variance
X_train = scalar.fit_transform(X_train) #scale the training data
X_test = scalar.transform(X_test) #scale the testing data using the same transformation as for the training data

#Neural Network
start_timeNN = time.time()  # Record the current time
classifier_NN = MLPClassifier((8, 12, 10), max_iter = 5000, solver = 'sgd', alpha = 0.1, activation = 'relu', learning_rate='adaptive', random_state=42) #create a multi-layer perceptron classifier with 3 hidden layers, with 16, 24, and 4 nodes respectively, using stochastic gradient descent as the solver, 0.1 as the L2 penalty parameter, ReLU as the activation function, and adaptive learning rate for weight updates.
classifier_NN.fit(X_train, y_train) #fit the neural network classifier on the training data
y_pred_NN = classifier_NN.predict(X_test) #make predictions on the testing data using the trained neural network classifier
print("Accuracy on Neural Network: {0}".format(accuracy_score(y_test, y_pred_NN))) #print the accuracy score of the neural network classifier
end_timeNN = time.time()  # Record the current time again
execution_timeNN = end_timeNN - start_timeNN  # Calculate the difference to get the execution time
print("Execution time for NN: ", execution_timeNN, " seconds")

#Random Forest
start_timeRF = time.time()  # Record the current time
classifier_RF = RandomForestClassifier(n_estimators = 200, max_depth = 13, n_jobs= -1, random_state=42) #create a random forest classifier with 200 trees, a maximum depth of 13, using all available cores for parallel computation, and a random state of 42
classifier_RF.fit(X_train, y_train) #fit the random forest classifier on the training data
y_pred_RF = classifier_RF.predict(X_test) #make predictions on the testing data using the trained random forest classifier
print("Accuracy on Random Forest Classifier: {0}".format(accuracy_score(y_test, y_pred_RF))) #print the accuracy score of the random forest classifier
end_timeRF = time.time()  # Record the current time again
execution_timeRF = end_timeRF - start_timeRF  # Calculate the difference to get the execution time
print("Execution time for RF: ", execution_timeRF, " seconds")

