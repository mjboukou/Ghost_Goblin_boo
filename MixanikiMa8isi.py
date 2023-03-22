import tensorflow as tf
import matplotlib.pyplot as plt
import time
from IPython.display import Markdown, display
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import metrics
print(tf.__version__)
from keras.layers import Dense, Activation
from keras.models import Sequential


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, plot_confusion_matrix
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')
train_data.head()
train_data.describe()
train_data.isnull().sum()
train_data.shape
train_data.color.value_counts()
train_data.type.value_counts()

#Check for duplicates
train_data.duplicated().sum()
train_data.info()
train_data.drop(['id'], axis=1, inplace=True)
train_data
X = train_data.iloc[:,:-1].values
y = train_data.iloc[:,-1].values
print(X.shape)
print(y.shape)

#Splitting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)

from sklearn.preprocessing import LabelEncoder
labelencoderX = LabelEncoder()
X_train[:, 4] = labelencoderX.fit_transform(X_train[:, 4])
X_test[:, 4] = labelencoderX.transform(X_test[:, 4])
X_val[:, 4] = labelencoderX.transform(X_val[:, 4])


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

method = input("Enter the name of the method you want to use for classification (SVC | KNN | NN | NBC) : ")

if(method == 'SVC' or method =='svc' ):
	from sklearn.svm import SVC
	############
	# SVM Model#
	############
	start1 = time.time()
	krnl = str(input("Give the type of kernel you want to use (linear | gaussian ) : "))
	svc = SVC(C=1, kernel= krnl , gamma="auto")
	svc.fit(X_train, y_train)
	end1 = time.time()
	svm_time = end1-start1
	print("SVM Time: {:0.2f} minute".format(svm_time/60.0))
	# SVM report and analysis
	y_pred_svc = svc.predict(X_test)
	svc_f1 = metrics.f1_score(y_test, y_pred_svc, average= "weighted")
	svc_accuracy = metrics.accuracy_score(y_test, y_pred_svc)
	svc_cm = metrics.confusion_matrix(y_test, y_pred_svc)
	print("-----------------SVM Report---------------")
	print("F1 score: {}".format(svc_f1))
	print("Accuracy score: {}".format(svc_accuracy))
	print("Confusion matrix: \n", svc_cm)
	print('Plotting confusion matrix')
	plt.figure()
	plt.show()
	print(metrics.classification_report(y_test, y_pred_svc))
    
elif (method == 'KNN' or method =='knn' ):
    
    from sklearn.neighbors import KNeighborsClassifier
    ############
    # KNN Model#
    ############
    k = int(input("Give the neighbors for the KNN classifier : "))
    mtrc = str("euclidean")
    start2 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k , weights='distance' , metric= mtrc)
    knn.fit(X_train, y_train) 
    y_pred_knn = knn.predict(X_test)
    end2 = time.time()
    knn_time = end2-start2
    print("KNN Time: {:0.2f} minute".format(knn_time/60.0))
    # KNN report and analysis
    knn_f1 = metrics.f1_score(y_test, y_pred_knn, average= "weighted")
    knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn)
    knn_cm = metrics.confusion_matrix(y_test, y_pred_knn)
    print("-----------------K-nearest neighbors Report---------------")
    print("F1 score: {}".format(knn_f1))
    print("Accuracy score: {}".format(knn_accuracy))
    print("Confusion matrix: \n", knn_cm)
    print('Plotting confusion matrix')
    plt.figure()
    plt.show()
    print(metrics.classification_report(y_test, y_pred_knn))

elif (method == 'NN' or method == 'nn'):
    
    ############
    # NN MODEL #
    ############
    hiden_layers = input("Give the number of Hidden Layers of the Nural Netwok 1 or 2: ")
    if (hiden_layers == '1'):
        K = input("Give of the Hidden Neurals 50 or 100 or 200 : ")
        model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(X_train.shape)),
        tf.keras.layers.Dense(K, activation='sigmoid'), # Layer one of K neuronswith sigmoid activation function
        ])
    elif(hiden_layers == '2'):
        K1 = input("Give the number of the hidden neurals 50 or 100 or 200")
        K2 = int(input("Give the number of the hidden neurals 25 or 50 or 100"))
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(X_train.shape)),
        tf.keras.layers.Dense(K1, activation='sigmoid'), # Layer one of K1 neurons with sigmoid activation function
        tf.keras.layers.Dense(K2, activation='sigmoid'), # Layer two of K2 neurons with sigmoid activation function
        ])
        
    model.compile(optimizer='sgd',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(X_train=np.asarray(X_train).astype(np.float32), y_train= np.asarray(y_train).astype(np.float32),epochs=10)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("-----------------NEURAL NETWORK---Report---------------")
    print('\nTest accuracy:', test_acc)
    propability_model = tf.keras.Sequential([model,tf.keras.layers.Dense(10),tf.keras.layers.Softmax(),])