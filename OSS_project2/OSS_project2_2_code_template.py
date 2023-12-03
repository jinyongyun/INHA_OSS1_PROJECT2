import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def sort_dataset(dataset_df):
	#TODO:  Sort the entire data by year(해당 시즌) column in ascending order
   df_sorted = dataset_df.sort_values(by='year', ascending=True)
   return df_sorted

def split_dataset(dataset_df):	
	#TODO: performs data preprocessing by rescaling the salary column and then splits the dataset into training and testing sets, separating the features and target variable accordingly. The resulting sets are suitable for training and evaluating regression models, where the goal is to predict the salary based on the given features.

    dataset_df['salary'] *= 0.001

    train = dataset_df.loc[:1717]
    test = dataset_df.loc[1718:]
    
    X_train = train.drop('salary', axis=1)
    Y_train = train['salary']
    X_test = test.drop('salary', axis=1)
    Y_test = test['salary']

    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
	#TODO:   to filter the input DataFrame and retain only the columns that are expected to contain numerical data, discarding any non-numerical columns. This can be useful when you want to focus on the numerical features for analysis or modeling purposes.
    numerical_column = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 
                           'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    numerical_columns = dataset_df[numerical_column]
    return numerical_columns

def train_predict_decision_tree(X_train, Y_train, X_test):
	#TODO: encapsulates the entire process of training a Decision Tree Regressor model on a given training set and using the trained model to predict target values for a separate test set. Decision Trees are a type of supervised learning algorithm that can be used for both classification and regression tasks. They work by recursively partitioning the feature space into regions and assigning a value to each region based on the average (or another criterion) of the training data in that region.
    dt_cls = DecisionTreeRegressor()
    dt_cls.fit(X_train, Y_train)
    return dt_cls.predict(X_test)
 

def train_predict_random_forest(X_train, Y_train, X_test):
	#TODO:  this function encapsulates the entire process of training a Random Forest Regressor model on a given training set and using the trained model to predict target values for a separate test set. Random Forest is an ensemble learning method that combines the predictions of multiple decision trees to improve overall accuracy and robustness. In a regression context, it is used to predict continuous numerical outcomes.
 rf_cls = RandomForestRegressor()
 rf_cls.fit(X_train, Y_train)
 return rf_cls.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
	#TODO:  encapsulates the process of training an SVR model with standardized features (using StandardScaler) and using the trained model to predict target values for a separate test set. Support Vector Regression is a regression algorithm that aims to find a hyperplane that minimizes the error while staying within a specified margin around the predicted values. It is particularly useful when dealing with non-linear relationships between features and the target variable.
    svm_pipe = make_pipeline(
	 StandardScaler(), 
     SVR()
    )
    svm_pipe.fit(X_train, Y_train)
    return svm_pipe.predict(X_test)
 

def calculate_RMSE(labels, predictions):
	#TODO:  calculate the Root Mean Squared Error (RMSE) between the predicted values and the actual labels. RMSE is a common metric used to evaluate the accuracy of regression models. It measures the average magnitude of the differences between predicted and actual values, giving more weight to larger differences.
   rmse = np.sqrt(np.mean((predictions-labels)**2))
   return rmse

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))