# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:21:51 2020

@author: iman
"""


import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

VERSION_NUM = "5.0"

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data set from CSV file
data_df = pd.read_csv("train_data_cpe_V" + VERSION_NUM + ".csv")

# Preprocessing Data
print("Preprocessing ...")
preprocessingTime1= int(round(time.time() * 1000))

data_df["AV"] = data_df["AV"].str.replace('N','0')
data_df["AV"] = data_df["AV"].str.replace('A','1')
data_df["AV"] = data_df["AV"].str.replace('L','2')
data_df["AV"] = data_df["AV"].str.replace('P','3')

data_df["AC"] = data_df["AC"].str.replace('L','0')
data_df["AC"] = data_df["AC"].str.replace('H','1')

data_df["PR"] = data_df["PR"].str.replace('N','0')
data_df["PR"] = data_df["PR"].str.replace('L','-1')
data_df["PR"] = data_df["PR"].str.replace('H','1')

data_df["UI"] = data_df["UI"].str.replace('N','0')
data_df["UI"] = data_df["UI"].str.replace('R','1')

data_df["Scope"] = data_df["Scope"].str.replace('U','0')
data_df["Scope"] = data_df["Scope"].str.replace('C','1')

data_df["CI"] = data_df["CI"].str.replace('N','0')
data_df["CI"] = data_df["CI"].str.replace('L','-1')
data_df["CI"] = data_df["CI"].str.replace('H','1')

data_df["II"] = data_df["II"].str.replace('N','0')
data_df["II"] = data_df["II"].str.replace('L','-1')
data_df["II"] = data_df["II"].str.replace('H','1')

data_df["AI"] = data_df["AI"].str.replace('N','0')
data_df["AI"] = data_df["AI"].str.replace('L','-1')
data_df["AI"] = data_df["AI"].str.replace('H','1')

print("Preprocessing is done\tTime consumed:{}", int(round(time.time() * 1000)) - preprocessingTime1)

# Seperating X_set and Y_set
# Headers in order are: AV, AC, PR, UI, Scope, CI, II, AI, baseScore
X_set = data_df.drop('baseScore', axis=1).values
y_set = data_df[['baseScore']].values

# Splitting train and test set and changing X_set dtype to float64
X_train, X_test, y_train, y_test = train_test_split(np.array(X_set, dtype=np.float), y_set, test_size=0.25, shuffle=True)

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_train = X_scaler.fit_transform(X_train)
y_scaled_train = y_scaler.fit_transform(y_train)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_test = X_scaler.transform(X_test)
y_scaled_test = y_scaler.transform(y_test)

# Define model parameters
RUN_NAME = "run 1 v"+ VERSION_NUM +" 70_100_60 nodes"
learning_rate = 0.001
training_epochs = 100

# Define how many inputs and outputs are in our neural network
number_of_inputs = 8
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 70
layer_2_nodes = 100
layer_3_nodes = 60

# Reseting Kernel
tf.reset_default_graph()

# Section One: Define the layers of the neural network itself

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:
    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    training_writer = tf.summary.FileWriter('./logsv' + VERSION_NUM + '/training', session.graph)
    testing_writer = tf.summary.FileWriter('./logsv' + VERSION_NUM + '/testing', session.graph)

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_train, Y: y_scaled_train})

        # Create log file writers to record training progress.
        # We'll store training and testing log data separately.
        training_writer = tf.summary.FileWriter("./logsv" + VERSION_NUM + "/{}/training".format(RUN_NAME), session.graph)
        testing_writer = tf.summary.FileWriter("./logsv" + VERSION_NUM + "/{}/testing".format(RUN_NAME), session.graph)

        # Every 5 training steps, log our progress
        if epoch % 5 == 0:
            # Get the current accuracy scores by running the "cost" operation on the training and test data sets
            training_cost, training_summary = session.run([cost, summary],
                                                          feed_dict={X: X_scaled_train, Y: y_scaled_train})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_test, Y: y_scaled_test})

            # Write the current training status to the log files (Which we can view with TensorBoard)
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            # Print the current training status to the screen
            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))

    # Training is now complete!

    # Get the final accuracy scores by running the "cost" operation on the training and test data sets
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_train, Y: y_scaled_train})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_test, Y: y_scaled_test})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    # Now that the neural network is trained, let's use it to make predictions for our test data.
    # Pass in the X testing data and run the "prediciton" operation
    Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_test})

    # Unscale the data back to it's original units (dollars)
    Y_predicted = y_scaler.inverse_transform(Y_predicted_scaled)




#   Function to calculate accuracy
def CalculateAccuracy(y_test, Y_predicted): 
    
    #   Mean Absolute Error (MAE)
    MAE = mean_absolute_error(y_test, Y_predicted)
    
    #   Correlation Coefficient of Regression (r^2)
    CCoR = r2_score(y_test, Y_predicted)
    
    #   Mean Squared Error
    MSE = mean_squared_error(y_test, Y_predicted)
    
    measurements = "The Mean Absolute Error: " + str(MAE) + "\nCorrelation Coefficient of Regression (r²): " + str(CCoR) + "\nMean Squared Error : " + str(MSE)
    
    return measurements


measurements = CalculateAccuracy(y_test, Y_predicted)
print("\n"+measurements)


diff = []
index = []
for i in range(len(Y_predicted)):
    diff = diff + [Y_predicted[i]-y_test[i]]
    index = index + [i]


plt.plot(index, y_test, marker='', color='red', linewidth=0.5, alpha = 0.5, label = 'actual values')
plt.plot(index, Y_predicted, marker='', color='blue', linewidth=0.5, alpha = 0.2, label = 'prediction values')

# line1 = plt.scatter(index, y_test,color='black', label = 'actual')
# line1 = plt.scatter(index, Y_predicted, color='green', label = 'predictions')


# plt.scatter(Y_predicted, y_test,color='red', label = 'prediction/actual')

plt.axis(aspect='equal')
plt.xlabel('Common Vulnerabilities and Exposures (CVE)')
plt.ylabel('Base Score')
plt.legend(fontsize = 'large')
plt.title(measurements);
plt.show()
