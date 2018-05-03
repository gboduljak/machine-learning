# coding: utf-8

# In[1]:

from plot_history import plot_history
from roc_auc_callback import RocAucCallback
from keras.callbacks import *
from sklearn.metrics import roc_auc_score

# In[2]:

import math
from keras.callbacks import LearningRateScheduler

initialLr = 0.001
dropRate = 0.15


def step_decay(epoch):
    lrRates = []
    lr = initialLr * math.pow(2, -dropRate * epoch)
    lrRates.append(lr)
    return lr


lrScheduler = LearningRateScheduler(step_decay)


# In[3]:

def train_with_cv(model, batchSize=32, epochs=32, rocEvery=2, patience=10, shouldValidate=True):
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')

    earlyStopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=patience,
        verbose=0,
        mode='auto'
    )

    rocAuc = RocAucCallback(training_data=(X_train, y_train), validation_data=(X_val, y_val), runEvery=rocEvery)
    reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, cooldown=10, mode='auto')
    modelCheckpoint = ModelCheckpoint('/model.h5', monitor='val_loss')

    if (shouldValidate):
        return model.fit(
            X_train,
            y_train,
            batch_size=batchSize,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[lrScheduler, reduceLr, earlyStopping, modelCheckpoint, rocAuc]
        )
    else:
        return model.fit(
            X_train,
            y_train,
            batch_size=batchSize,
            epochs=epochs,
            callbacks=[lrScheduler, reduceLr]
        )

def evaluate_on_test(model):
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    metrics = model.evaluate(x = X_test, y = y_test, batch_size = 32)
    rocAuc = roc_auc_score(y_test, model.predict(X_test))

    return metrics, rocAuc

# In[4]:

import pandas as pd

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

currentPrediction = 0
loPrediction = 0
hiPrediction = 0
predictionModel = {}
predictionModelName = 'model'


def submission_on_epoch_end(epoch, logs):
    global currentPrediction
    global loPrediction
    global hiPrediction
    global labels
    global predictionModel
    global predictionModelName

    if currentPrediction >= loPrediction and currentPrediction <= hiPrediction:
        print('Predicting on submission...\n')
        submissionFilename = '{}-{}.csv'.format(predictionModelName, currentPrediction)
        submissionFrame = pd.read_csv('sample_submission.csv')
        predictions = predictionModel.predict(X_submission, batch_size=32)
        submissionFrame[labels] = predictions
        submissionFrame.to_csv(submissionFilename, index=False)
        print('Done.\n')

    currentPrediction = currentPrediction + 1

predictSubmissionCallback = LambdaCallback(on_epoch_end=submission_on_epoch_end)

# In[5]:

def train_with_submitting(model, epochs = 16, predictAfter = 5, predictBefore = 10):
    global currentPrediction
    global predictionModel
    global loPrediction
    global hiPrediction
    global labels

    X_train_full = np.load('X_train_full.npy')
    y_train_full = np.load('y_train_full.npy')

    currentPrediction = 0
    predictionModel = model
    loPrediction = predictAfter
    hiPrediction = predictBefore

    modelCheckpoint = ModelCheckpoint('/model.h5', monitor='loss')
    tensorboard = TensorBoard('/model/tensorboard/', embeddings_layer_names=['embedding_1'])

    return predictionModel.fit(
        X_train_full,
        y_train_full,
        batch_size=32,
        epochs=epochs,
        callbacks=[modelCheckpoint, tensorboard, predictSubmissionCallback]
    )