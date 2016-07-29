
#Watson Health Hackathon

# required libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation


# from MAC; different path for the data
inputFilePath = '/Users/feraselzarwi/PycharmProjects/untitled/Hackathon/'
inputFileName = 'weatherdata.csv'

df = pd.read_csv(open(inputFilePath + inputFileName, 'rb'))

# check the first few rows
df.head()
df.columns
df.shape
df.hist() # looking at histograms for all covariates in the data file

# let's reshuffle the rows and take a small subsample to work with
num =  df.shape[0]#number of samples
df1 = df.loc[random.sample(list(df.index),num)]

# split the data into training and validation/hold-out set (85% versus 15%)
data = np.array(df1)
data_training = data[0:0.85*data.shape[0],:]
data_validation = data[0.85*data.shape[0]:,:]

# determining the covariates and regressor values for training and validation sets
data_training_covariates =data_training[:,1:5]
data_training_labels = data_training[:,5]

data_validation_covariates = data_validation[:, 1:5]
data_validation_labels = data_validation[:,5]

# Machine Learning Method #1
# Random Forest Regression

#determining optimal number of trees in forest to minimize MSE (mean sqaured error) on the validation set

MSE_RF = np.zeros((13,1))
count = 0
for n in [5, 10, 20,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500 ]:
    forest = RandomForestRegressor(n_estimators = n)
    #forest = forest.fit(data_training_covariates, data_training_labels)
    RF_score = cross_validation.cross_val_score(forest, data_training_covariates, data_training_labels, cv=10,
                                                scoring='mean_squared_error')
    score = np.mean(RF_score)
    MSE_RF[count] = -1 * score
    count = count + 1

# plots
plt.title('Cross Validation MSE Plot ')
num_trees = [5, 10, 20,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500 ]
line1, = plt.plot(num_trees, MSE_RF)
plt.ylabel('MSE')
plt.xlabel('Number of Trees in Forest')
plt.show()

optimal_num_trees = 150
# let's play with tree depth

MSE_RF_2 = np.zeros((20,1))
count = 0
for n in [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20]:
    forest = RandomForestRegressor(n_estimators = 150,max_depth = n )
    #forest = forest.fit(data_training_covariates, data_training_labels)
    RF_score_2 = cross_validation.cross_val_score(forest, data_training_covariates, data_training_labels, cv=10,
                                                scoring='mean_squared_error')
    score = np.mean(RF_score_2)
    MSE_RF_2[count] = -1 * score
    count = count + 1



plt.title('Cross Validation MSE Plot ')
depth = [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20]
line1, = plt.plot(depth, MSE_RF_2)
plt.ylabel('MSE')
plt.xlabel('Depth of Trees, Num Trees = 150')
plt.show()


optimal_depth = 3
optimal_num_trees=150


forest = RandomForestRegressor(n_estimators = 150,max_depth = 3 )
forest = forest.fit(data_training_covariates, data_training_labels)
output = forest.predict(data_validation_covariates)
MSE_RF_final = sum((output - data_validation_labels) ** 2)/data_validation_labels.shape[0] # computing the relative MSE

# Decision Tree
# determining the optimal depth of the tree via cross validation

MSE_DT = np.zeros((20,1))
count = 0
for n in [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20]:
    decision_tree = tree.DecisionTreeRegressor(max_depth = n)
    #decision_tree.fit(data_training_covariates, data_training_labels)
    DT_score = cross_validation.cross_val_score(decision_tree, data_training_covariates, data_training_labels, cv=10,
                                                 scoring='mean_squared_error')
    score = np.mean(DT_score)
    MSE_DT[count] = -1*score
    count = count + 1

# plots
plt.title('Cross Validation MSE Plot ')
depth = [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20]
line1, = plt.plot(depth, MSE_DT, label="")
plt.ylabel('MSE')
plt.xlabel('Depth of Tree')
plt.show()


optimal_depth = 3


# fit a decision tree using training data and check performance on validation data
decision_tree = tree.DecisionTreeRegressor(max_depth = 3)
decision_tree.fit(data_training_covariates, data_training_labels)
output = decision_tree.predict(data_validation_covariates)
MSE_DT_final = sum((output - data_validation_labels) ** 2)/data_validation_labels.shape[0]

# OLS regression --> endogeneity is a major concern!!

# K Nearest Neighbor

# first we need to normalize our data and metric is the Euclidean distance (L2 norm)
#One important point that we have to keep in mind is that if we used any normalization
#or transformation technique on our training dataset, we’d have to use the same parameters
#on the test dataset and new unseen data.

data_training_covariates_mean = np.mean(data_training_covariates, axis=0)
data_training_covariates_std = np.var(data_training_covariates, axis=0)**0.5


data_training_covariates_NN = (data_training_covariates - data_training_covariates_mean)/data_training_covariates_std
data_training_labels_NN = data_training_labels


data_validation_covariates_NN = (data_validation_covariates - data_training_covariates_mean)/data_training_covariates_std
data_validation_labels_NN = data_validation_labels

# determining the optimal value of K through cross validation

knn = KNeighborsRegressor()
KNeighborsRegressor(n_neighbors=2, p=2, weights='uniform')

knn_score = cross_validation.cross_val_score(knn,data_training_covariates_NN, data_training_labels_NN, cv=10, scoring='mean_squared_error')
score = np.mean(knn_score)
MSE_NN = -1*score

MSE_NN = np.zeros((33,1))
count = 0
for n in [2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,25, 30, 35,40, 50, 55, 60, 70, 80, 90,100, 110, 120,130]:
    knn = KNeighborsRegressor(n_neighbors=n, p=2, weights='uniform')
    #knn.fit(data_training_covariates_NN, data_training_labels_NN)

    knn_score = cross_validation.cross_val_score(knn, data_training_covariates_NN, data_training_labels_NN, cv=10,
                                                 scoring='mean_squared_error')
    score = np.mean(knn_score)
    MSE_NN[count] = -1 * score
    count = count + 1

# plots
plt.title('Cross Validation MSE Plot ')
neigh = [2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,25, 30, 35,40, 50, 55, 60, 70, 80, 90,100, 110, 120,130]
line1, = plt.plot(neigh, MSE_NN)
plt.ylabel('MSE')
plt.xlabel('Number of Neighbors')
plt.show()


# optimal K from cross validation results
optimal_k = 55


# using those values to predict on the validaiton set !
model_NN = KNeighborsRegressor(n_neighbors=55, p=2, weights='uniform')
model_NN.fit(data_training_covariates_NN, data_training_labels_NN)

output = model_NN.predict(data_validation_covariates_NN)

MSE_NN_final = sum((output - data_validation_labels_NN) ** 2)/data_validation_labels_NN.shape[0]


# final predictor to be used in the app!!!

# now let us train using all data points in our database

data_final = np.array(df1)
data_training_covariates_final =data_final[:,1:5]
data_training_labels_final = data_final[:,5]

def predictor(data_training_covariates_final, data_training_labels_final, new_data_covariates):
    forest = RandomForestRegressor(n_estimators=150, max_depth=3)
    forest = forest.fit(data_training_covariates_final, data_training_labels_final)
    output = forest.predict(new_data_covariates)
    return output