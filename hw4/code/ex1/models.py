import os, sys
import random as rd
import pandas as pd
import numpy as np
from numpy import loadtxt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


#number of nodes in each layer
h1 = [1, 4, 12]
h2 = [0, 3]

# initialize a function to create model, required for KerasClassifier
def make_model(h1, h2):
    # create model
    model = Sequential()
    # add input layer with input_shape = (2,) with tanh activation
    model.add(Dense(h1, input_shape=2, activation='tanh'))
    # add the first hidden layer with tanh activation
    if (h2 != 0):
        model.add(Dense(h2, activation='tanh'))
    # output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def fit_model(dataset, h1, h2):
    # split into input (X) and output (y) variables
    X = dataset[['x1', 'x2']]
    X = X.to_numpy()
    y = dataset['labels']
    y = y.to_numpy()
    
    # create model
    model = make_model(h1, h2)
    # generate 10-fold cross validation
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    accuracy_lst = []
    balanced_accuracy_lst = []
    auroc_lst = []
    #for each fold, train the model and evaluate the model
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit the model
        model.fit(X_train, y_train, epochs=100, verbose=0)
        # evaluate the model
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracy_lst.append(accuracy)
        # get the predictions
        y_pred = model.predict(X_test)
        #compute the balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracy_lst.append(balanced_accuracy)
        # area under the ROC curve
        auroc = roc_auc_score(y_test, model.predict_proba(X_test))
        auroc_lst.append(auroc)
    
    # return the mean of the accuracy, balanced accuracy, and auroc
    return np.mean(accuracy_lst), np.mean(balanced_accuracy_lst), np.mean(auroc_lst)

concepts = ['A', 'B']
n = [250, 1000, 10000]

results = pd.DataFrame(columns=['Noise', 'Concept', 'N', 'H1', 'H2', 'Accuracy', 'Balanced Accuracy', 'AUROC'])

for concept in concepts:
    for num_point in n:
        for h_1 in h1:
            for h_2 in h2:
                model = make_model(h_1, h_2)
                data = pd.read_csv(ROOT_DIR + '/data/concept_' + concept + '_' + str(num_point) + '.csv')
                
                accuracy, balanced_accuracy, auroc = fit_model(data, h_1, h_2)
                # put the results in a dataframe
                results.append({'Noise': 0, 'Concept': concept, 'N': num_point, 'H1': h_1, 'H2': h_2, 'Accuracy': accuracy, 'Balanced Accuracy': balanced_accuracy, 'AUROC': auroc}, ignore_index=True)
                
                #fit the model on the dataset w/o noise
                data_wo_noise = loadtxt(ROOT_DIR + '/data/no_noise/concept_' + concept + '_' + str(num_point) + '.csv', delimiter=',')
                accuracy, balanced_accuracy, auroc = fit_model(data_wo_noise, h_1, h_2)
                results.append({'Noise': 1, 'Concept': concept, 'N': num_point, 'H1': h_1, 'H2': h_2, 'Accuracy': accuracy, 'Balanced Accuracy': balanced_accuracy, 'AUROC': auroc}, ignore_index=True)