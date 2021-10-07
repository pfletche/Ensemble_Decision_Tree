import pickle
import time
import pandas as pd
import numpy as np
import algorithms as algs
from output_data import addNewDataToCsv
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Name the output file
outputFile = 'results'

# Get the ground truth image
ground_truth = Image.open('groundtruth.png')
pltt = ground_truth.getpalette()

# Get the class values from the image palette
ground_truth = ground_truth.convert('P', palette=pltt)
ground_truth = np.array(ground_truth.getdata())

# Get the image channels for training
channel_data = pd.read_csv('all_channels.csv', index_col=0)

# Print the shape of the channel data for verification
print('channel_data shape:', channel_data.shape)

# *** Uncomment the line below to drop the appropriate features for each trial ***
# channel_data = channel_data.drop(['dsm', 'transparent_reflectance_red edge', 'transparent_reflectance_nir'],axis=1)

# Create dataframe
DF = pd.DataFrame()

# Train Test Split - sets training data to 20% of total data set
print('Starting train test split')
X_train, X_test, y_train, y_test = train_test_split(channel_data, ground_truth,
                                                    train_size=0.2, test_size=0.8,
                                                    shuffle=True, random_state=22)

# Set the test values to the entire data set
X_test = channel_data
y_test = ground_truth

print('Finished train test split')

# Output Dataframe
output = pd.DataFrame()

# *******************************************************************************
# Decision Tree
# *******************************************************************************
startTime = time.time()
print('\n**** Starting Decision Tree ****\n')

predictions, finishTime = algs.runDecisionTree(X_train, X_test, y_train, y_test)

DTruntime = finishTime - startTime

print('Finished Decision Tree in ', finishTime - startTime)
addNewDataToCsv('Decision Tree', DTruntime, predictions, ground_truth, outputFile)

predictions = predictions.astype(np.uint8)
predicted_image = predictions.reshape(5123,8045)

predicted_image = Image.fromarray(predicted_image)
predicted_image.putpalette(pltt)
predicted_image.save('DT_predicted_image_' + outputFile + '.png')

# *******************************************************************************
# Bagging Classifier
# *******************************************************************************
startTime = time.time()
print('\n**** About to run Bagging Classifier ****\n')

predictions, finishTime = algs.runBaggingTree(X_train, X_test, y_train, y_test)

c_matrix = confusion_matrix(ground_truth, predictions)
print('Confusion Matrix', c_matrix)

save_confusion = open("confusion.pkl","wb")
pickle.dump(c_matrix, save_confusion)
save_confusion.close()

RFruntime = finishTime - startTime

print('Finished Bagging Classifier in', finishTime - startTime)
addNewDataToCsv('Bagging Classifier', RFruntime, predictions, ground_truth, outputFile)

predictions = predictions.astype(np.uint8)
predicted_image = predictions.reshape(6904,10623)

predicted_image = Image.fromarray(predicted_image)
predicted_image.putpalette(pltt)
predicted_image.save('BG_new_predicted_image_' + outputFile + '.png')
