# Studi-Algoritma-Deep-Learning-Untuk-Mendeteksi-Aritmia-Berdasarkan-Sinyal-Elektrokardiogram
# Arrhythmia Classification Tutorial

This will guide you through the process of classifying arrhythmia using neural networks. You'll learn how to process the data, build various neural network models (CNN, LSTM, GRU), and evaluate their performance.

## Prerequisites
- Basic knowledge of Python programming.
- Familiarity with neural networks and Keras library.
- Access to the MIT-BIH Arrhythmia Database.

## Tutorial Steps

Follow the steps below to understand the entire process:

### Step 1: Reading the Input Data

```python
import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import csv
import itertools
import collections

import pywt
from scipy import stats

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv1D, AvgPool1D, Flatten, Dense, Dropout, Softmax,LSTM,GRU
from tensorflow.keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import regularizers
```

### Step 2: Reading files

```python
!pip -q install wfdb==3.4.0
!wget -q https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
!unzip -qo /content/mit-bih-arrhythmia-database-1.0.0.zip

path = 'D:/.../mitbih_database/mitbih_database/'  
#input path
window_size = 180
maximum_counting = 10000

classes = ['N', 'L', 'R', 'A', 'V']
n_classes = len(classes)
count_classes = [0]*n_classes

X = list()
y = list()
```

### Step 3: Seperating files and annotations

```python
for f in filenames:
    filename, file_extension = os.path.splitext(f)
    
    if(file_extension == '.csv'):
        records.append(path + filename + file_extension)
    else:
        annotations.append(path + filename + file_extension) 
```

### Step 4: Data Denoising

```python
def denoise(data): 
    w = pywt.Wavelet('sym4') #syn4 - 4 max level decomposition
    maxlev = pywt.dwt_max_level(len(data), w.dec_len) #max useful decomposition
    threshold = 0.04 # filtering threshold

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev) #adding coefficients
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
        
    datarec = pywt.waverec(coeffs, 'sym4') #waverec
    
    return datarec
```

### Step 5: Reading R positions 

```python
filename = f'D:/1buatTA/mit-bih-arrhythmia-database-1.0.0/{str(100 + signal_number)}'
record = wfdb.rdrecord(filename, sampfrom=180, sampto=4000,)    
annotation = wfdb.rdann(filename, 'atr', sampfrom=180, sampto=4000,shift_samps=True)

wfdb.plot_wfdb(record=record, annotation=annotation, time_units='seconds',figsize=(15,8))
```

### Step 6: Class Distribution
```python
df = pd.DataFrame(X) #Array to df
per_class = df[df.shape[1]-1].value_counts() #class distribution
print(per_class)
```

### Step 7: Train-Test split

```python
train_x, test_x, train_y, test_y = train_test_split(rr_intervals, qrs_durations, test_size=0.2, random_state=42)
```

### Step 8: SMOTE

```python
#down sampling class 0
df_0=(train[train[train.shape[1]-1]==0]).sample(7000,random_state=42)

#up sampling class 1
df_1=train[train[train.shape[1]-1]==1]
df_1_upsample=resample(df_1,replace=True,n_samples=7000,random_state=23)

#up sampling class 2
df_2=train[train[train.shape[1]-1]==2]
df_2_upsample=resample(df_2,replace=True,n_samples=7000,random_state=23)

#up sampling class 3
df_3=train[train[train.shape[1]-1]==3]
df_3_upsample=resample(df_3,replace=True,n_samples=7000,random_state=23)

#up sampling class 4
df_4=train[train[train.shape[1]-1]==4]
df_4_upsample=resample(df_4,replace=True,n_samples=7000,random_state=23)

```

### Step 9: CNN Model

```python
model = Sequential()
  #convolution layer 1
  model.add(Conv1D(filters=16, kernel_size=13, padding='same', activation='relu',input_shape=(360, 1)))
  model.add(AvgPool1D(pool_size=3, strides=2))
  #convoulution layer 2
  model.add(Conv1D(filters=32, kernel_size=15, padding='same', activation='relu'))
  model.add(AvgPool1D(pool_size=3, strides=2))
  #convolution layer 3
  model.add(Conv1D(filters=64, kernel_size=17, padding='same', activation='relu'))
  model.add(AvgPool1D(pool_size=3, strides=2))
  #convolution layer 4
  model.add(Conv1D(filters=128, kernel_size=19, padding='same', activation='relu'))
  model.add(AvgPool1D(pool_size=3, strides=2))
  #flattening layer 
  model.add(Flatten())
  #Droupout threshold 0.5
  model.add(Dropout(0.5))
  #outer dense layer with 5 neurons represent 5 classes
  model.add(Dense(5,kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)))
  model.add(Softmax()) #probability of the classes
    
  model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
  model.summary()

  return model

cnn=cnn_model()
```

### Step 10: KFold CNN

```python
kf = KFold(5, shuffle=True, random_state=42)
oos_y = []
oos_pred = []

acc_per_fold = []
loss_per_fold = []
rmse_per_fold = []
conf_matrices = []  # List to store confusion matrices
fold = 0

for train, test in kf.split(train_x, train_y):
    fold += 1
    print(f"Fold #{fold}")

    x_train = train_x[train]
    y_train = train_y[train]
    x_test = train_x[test]
    y_test = train_y[test]

    history = cnn.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0, epochs=10)

    pred = cnn.predict(x_test)

    oos_y.append(y_test)
    oos_pred.append(pred)

    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    conf_matrices.append(cm)
    
    score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    rmse_per_fold.append(score)
    print(f"Fold score (RMSE): {score}")

    scores = cnn.evaluate(x_test, pred, verbose=0)
    print(f'Score for fold {fold}: {cnn.metrics_names[0]} of {scores[0]}; {cnn.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
```

### Step 11: Classification CNN

```python
y_true=[]
for element in test_y:
    y_true.append(np.argmax(element))
prediction_proba=cnn.predict(test_x)
prediction=np.argmax(prediction_proba,axis=1)

# Calculate the confusion matrix
model_cf_matrix = confusion_matrix(y_true, prediction)

# Calculate specificity for each class
specificity_per_class = []
for i in range(len(classes)):
    tn = np.sum(np.delete(model_cf_matrix, i, axis=0)[:, np.delete(np.arange(len(classes)), i)])
    fp = np.sum(model_cf_matrix[:, i]) - model_cf_matrix[i, i]
    specificity = tn / (tn + fp)
    specificity_per_class.append(specificity)
```

### Step 12: LSTM Model

```python
def get_lstm():
    lstm_model = Sequential()
    #one layer of LSTM
    lstm_model.add(LSTM(64, input_shape=(360,1)))
    #Fully connected dense layer
    lstm_model.add(Dense(128, activation = 'relu'))
    #dropout threshold 0.3
    lstm_model.add(Dropout(0.3))
    #outer dense layer with 5 neurons
    lstm_model.add(Dense(5, activation = 'softmax'))
    
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.summary()
    return lstm_model

lstm=get_lstm()
```

### Step 13: KFold LSTM

```python
kf = KFold(5, shuffle=True, random_state=42)
oos_y = []
oos_pred = []

lstm_acc_per_fold = []
lstm_loss_per_fold = []
lstm_rmse_per_fold = []
lstm_conf_matrices = []  # List to store confusion matrics
fold = 0

for train, test in kf.split(train_x, train_y):
    fold += 1
    print(f"Fold #{fold}")

    x_train = train_x[train]
    y_train = train_y[train]
    x_test = train_x[test]
    y_test = train_y[test]

    history = lstm.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0, epochs=10)

    pred = lstm.predict(x_test)

    oos_y.append(y_test)
    oos_pred.append(pred)

    # Compute the confusion matrix
    cm_lstm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    lstm_conf_matrices.append(cm_lstm)
    # print(f"Confusion matrix for Fold #{fold}:\n{cm_lstm}")
    
    score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    lstm_rmse_per_fold.append(score)
    print(f"Fold score (RMSE): {score}")

    scores = lstm.evaluate(x_test, pred, verbose=0)
    print(f'Score for fold {fold}: {lstm.metrics_names[0]} of {scores[0]}; {lstm.metrics_names[1]} of {scores[1] * 100}%')
    lstm_acc_per_fold.append(scores[1] * 100)
    lstm_loss_per_fold.append(scores[0])
```

### Step 14: Classification LSTM

```python
y_true=[]
for element in test_y:
    y_true.append(np.argmax(element))
prediction_proba=lstm.predict(test_x)
prediction=np.argmax(prediction_proba,axis=1)

# Calculate the confusion matrix
model_cf_matrix = confusion_matrix(y_true, prediction)

# Calculate specificity for each class
specificity_per_class = []
for i in range(len(classes)):
    tn = np.sum(np.delete(model_cf_matrix, i, axis=0)[:, np.delete(np.arange(len(classes)), i)])
    fp = np.sum(model_cf_matrix[:, i]) - model_cf_matrix[i, i]
    specificity = tn / (tn + fp)
    specificity_per_class.append(specificity)
```

### Step 15: GRU Model

```python
def get_gru():
    gru_model = Sequential()
    #single layer GRU
    gru_model.add(GRU(64, input_shape=(360,1)))
    #fully connected outer layer
    gru_model.add(Dense(128, activation = 'relu'))
    #droput threshold 0.3
    gru_model.add(Dropout(0.3))
    #outer layer with 5 neurons
    gru_model.add(Dense(5, activation = 'softmax'))
    
    gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    gru_model.summary()
    return gru_model

gru=get_gru()
```

### Step 16: KFold GRU

```python
kf = KFold(5, shuffle=True, random_state=42)
oos_y = []
oos_pred = []

gru_acc_per_fold = []
gru_loss_per_fold = []
gru_rmse_per_fold = []
gru_conf_matrices = []  # List to store confusion matrics
fold = 0

for train, test in kf.split(train_x, train_y):
    fold += 1
    print(f"Fold #{fold}")

    x_train = train_x[train]
    y_train = train_y[train]
    x_test = train_x[test]
    y_test = train_y[test]

    history = gru.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0, epochs=10)

    pred = gru.predict(x_test)

    oos_y.append(y_test)
    oos_pred.append(pred)

    # Compute the confusion matrix
    cm_gru = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    gru_conf_matrices.append(cm_gru)
    # print(f"Confusion matrix for Fold #{fold}:\n{cm_gru}")
    
    score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    gru_rmse_per_fold.append(score)
    print(f"Fold score (RMSE): {score}")

    scores = gru.evaluate(x_test, pred, verbose=0)
    print(f'Score for fold {fold}: {gru.metrics_names[0]} of {scores[0]}; {gru.metrics_names[1]} of {scores[1] * 100}%')
    gru_acc_per_fold.append(scores[1] * 100)
    gru_loss_per_fold.append(scores[0])

```

### Step 17: Classification GRU

```python
y_true=[]
for element in test_y:
    y_true.append(np.argmax(element))
prediction_proba=gru.predict(test_x)
prediction=np.argmax(prediction_proba,axis=1)

# Calculate the confusion matrix
model_cf_matrix = confusion_matrix(y_true, prediction)

# Calculate specificity for each class
specificity_per_class = []
for i in range(len(classes)):
    tn = np.sum(np.delete(model_cf_matrix, i, axis=0)[:, np.delete(np.arange(len(classes)), i)])
    fp = np.sum(model_cf_matrix[:, i]) - model_cf_matrix[i, i]
    specificity = tn / (tn + fp)
    specificity_per_class.append(specificity)

# Print classification report
print("Classification Report:")
print(classification_report(y_true, prediction, target_names=classes, digits=4))

# Print specificity for each class
for i, class_name in enumerate(classes):
    print(f"Specificity for class {class_name}: {specificity_per_class[i]:.4f}")
```

## Conclusion
By following this tutorial, you should have a good understanding of how to process ECG data and apply neural networks for arrhythmia classification. Experiment with the models, adjust their parameters, and try other architectures to further enhance performance.

## Resources
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [Keras Documentation](https://keras.io/)
- [Python Official Documentation](https://docs.python.org/3/)
