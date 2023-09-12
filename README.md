# Studi-Algoritma-Deep-Learning-Untuk-Mendeteksi-Aritmia-Berdasarkan-Sinyal-Elektrokardiogram
# Arrhythmia Classification Tutorial

This will guide you through the process of classifying arrhythmia using neural networks. You'll learn how to process the data, build various neural network models (CNN, LSTM, GRU), and evaluate their performance.

## Prerequisites
- Basic knowledge of Python programming.
- Familiarity with neural networks and Keras library.
- Access to the MIT-BIH Arrhythmia Database.

## Tutorial Steps

Follow the steps below to understand the entire process:

### Step 1: # Reading the Input Data
# Reading the Input Data

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
##Reading files

```python
!pip -q install wfdb==3.4.0
!wget -q https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
!unzip -qo /content/mit-bih-arrhythmia-database-1.0.0.zip 
```

### Step 3: #Seperating files and annotations
#Seperating files and annotations

```python
for f in filenames:
    filename, file_extension = os.path.splitext(f)
    
    if(file_extension == '.csv'):
        records.append(path + filename + file_extension)
    else:
        annotations.append(path + filename + file_extension) 
```

### Step 4: # Data Denoising
# Data Denoising

```python
path = 'D:/1buatTA/mitbih_database/mitbih_database/'  
#input path
window_size = 180
maximum_counting = 10000

classes = ['N', 'L', 'R', 'A', 'V']
n_classes = len(classes)
count_classes = [0]*n_classes

X = list()
y = list()
```

### Step 5: #Reading R positions and Arrhythmia class
#Reading R positions and Arrhythmia class

```python
# Readint input files
filenames = next(os.walk(path))[2]

# Splitting the csv files and annotation files
records = list()
annotations = list()
filenames.sort()
```

### Step 6: #Distribusi Kelas
#Distribusi Kelas

```python
# grouping csv and annotation files
for f in filenames:
    filename, file_extension = os.path.splitext(f)
    
    # *.csv
    if(file_extension == '.csv'):
        records.append(path + filename + file_extension)

    # *.txt
    else:
        annotations.append(path + filename + file_extension)
```

### Step 7: #Train-Test split
#Train-Test split

```python
print(len(records))
```

### Step 8: #Sampling Train Data
#Sampling Train Data

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

### Step 9: #Independent and Dependent variables in Test Data
#Independent and Dependent variables in Test Data 

```python
# for each records
for r in range(0,len(records)):
    signals = []
    

    with open(records[r], 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read ECG data from .csv
        row_index = -1
        for row in spamreader:
            if(row_index >= 0):
                signals.insert(row_index, int(row[1]))
            row_index += 1
            
  #example plot of a sample 105.csv
    if r == 5:
        # Plot each patient's signal
        plt.title(records[5] + " Wave")
        plt.plot(signals[0:500])
        #plt.grid(False)
        plt.show()
        
    signals = denoise(signals) #denoising the signal
    # plotting after denoising
    if r == 5:
        plt.title(records[5] + " wave after denoised")
        plt.plot(signals[0:500])
        plt.show()
        
    signals = stats.zscore(signals) #performing Z score normalisation
    # Plot an example to the signals
    if r == 5:
        plt.title( records[5] + " wave after z-score normalization" )
        plt.plot(signals[0:500])
        plt.show()
    
    # Reading the R peaks and the annotation classes for each sample
    example_beat_printed = False
    with open(annotations[r], 'r') as fileID:
        data = fileID.readlines() 
        beat = list()

        for d in range(1, len(data)): 
            splitted = data[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted) 
            pos = int(next(splitted))
            arrhythmia_type = next(splitted) # arrhythmia clas
            if(arrhythmia_type in classes):
                arrhythmia_index = classes.index(arrhythmia_type)

                count_classes[arrhythmia_index] += 1
                if(window_size <= pos and pos < (len(signals) - window_size)):
                    beat = signals[pos-window_size:pos+window_size]  #R peaks  
                     
                    if r == 5 and not example_beat_printed: 
                        plt.title("A Beat from " + records[5] + " Wave")
                        plt.plot(beat)
                        plt.show()
                        example_beat_printed = True

                    X.append(beat)
                    y.append(arrhythmia_index)

print(np.shape(X), np.shape(y))

```

### Step 10: #Independent and dependent varibale in Training Data
#Independent and dependent varibale in Training Data

```python
signal_number = 5 
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = f'D:/1buatTA/mit-bih-arrhythmia-database-1.0.0/{str(100 + signal_number)}'
record = wfdb.rdrecord(filename, sampfrom=180, sampto=4000,)    
annotation = wfdb.rdann(filename, 'atr', sampfrom=180, sampto=4000,shift_samps=True)

wfdb.plot_wfdb(record=record, annotation=annotation, time_units='seconds',figsize=(15,8))
```

### Step 11: #CNN Model
#CNN Model

```python
# Load ECG data from PhysioNet (MIT-BIH Arrhythmia Database)
record_number = 105

filename = f'D:/1buatTA/mit-bih-arrhythmia-database-1.0.0/{str(record_number)}'
record = wfdb.rdrecord(filename)
ecg_signal = record.p_signal[:, 0]
fs = record.fs

# Limit the ECG signal to the first 10 seconds
end_time = 10
ecg_signal = ecg_signal[:int(fs * end_time)]

# Pan-Tompkins QRS Detection Algorithm
def pan_tompkins_qrs_detection(ecg_signal, fs):
    # Bandpass filter the ECG signal (0.5-50 Hz)
    low_cutoff = 0.5
    high_cutoff = 50
    nyquist_freq = 0.5 * fs
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)

    # Differentiate the filtered signal
    diff_ecg = np.diff(filtered_ecg)

    # Square the differentiated signal
    squared_ecg = diff_ecg**2

    # Moving-window integration (width: 150 ms, 200 ms for 360 Hz and 250 Hz respectively)
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width), 'valid')

    # Find R-peaks (QRS complexes)
    r_peaks, _ = find_peaks(integrated_ecg, distance=0.6 * fs)

    return r_peaks

# Perform QRS detection
qrs_indices = pan_tompkins_qrs_detection(ecg_signal, fs)

# Plot the ECG signal with detected QRS complexes
plt.figure(figsize=(12, 6))
time_axis = np.arange(0, len(ecg_signal)) / fs
plt.plot(time_axis, ecg_signal, color='blue')
plt.plot(time_axis[qrs_indices], ecg_signal[qrs_indices], 'ro', markersize=5)

plt.xlabel('Time (s)')
plt.ylabel('MLII (mV)')
plt.title('ECG Signal with QRS Detection')
plt.grid(True)
plt.show()




```

### Step 12: ##CNN model plot
##CNN model plot

```python
for i in range(0,len(X)):
        X[i] = np.append(X[i], y[i])

print(np.shape(X)) #combining X and y to a single file
```

### Step 13: ##KFold cross validation
##KFold cross validation

```python
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Function to calculate RR intervals
def calculate_rr_intervals(ecg_signal, fs):
    # Find R-peaks in the ECG signal
    r_peaks, _ = find_peaks(ecg_signal, distance=fs*0.6)
    
    # Calculate RR intervals (in seconds)
    rr_intervals = np.diff(r_peaks) / fs
    
    return rr_intervals

# Load ECG data from PhysioNet (MIT-BIH Arrhythmia Database)
record_number = 105

filename = f'D:/1buatTA/mit-bih-arrhythmia-database-1.0.0/{str(record_number)}'
record = wfdb.rdrecord(filename)
ecg_signal = record.p_signal[:, 0]
fs = record.fs

# Calculate RR intervals
rr_intervals = calculate_rr_intervals(ecg_signal, fs)

# Count the number of RR intervals
num_rr_intervals = len(rr_intervals)

print(f"Number of RR intervals: {num_rr_intervals}")

```

### Step 14: ##Training data Evaluation summary
##Training data Evaluation summary

```python
df = pd.DataFrame(X) #Array to df
per_class = df[df.shape[1]-1].value_counts() #class distribution
print(per_class)

#visualizing class distribution
plt.figure(figsize=(5,5))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(per_class, labels=['N', 'L', 'R', 'A', 'V'], colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
```

### Step 15: ##CNN Test Data Evaluation
##CNN Test Data Evaluation

```python
train,test=train_test_split(df,test_size=0.20) #train test split from the main dataframe
print("train : ", np.shape(train)) #train data shape
print("test  : ", np.shape(test)) #test data shape
```

### Step 16: ###Test Loss and Accuracy
###Test Loss and Accuracy

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

### Step 17: ###Test Data classification report
###Test Data classification report

```python
train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
#combining the sampled classes 
```

### Step 18: ###Test Data Confusion Matrix
###Test Data Confusion Matrix

```python
per_class = train_df[train_df.shape[1]-1].value_counts()
print(per_class)
#visualizing the sampled training data
plt.figure(figsize=(5,5))
my_circle=plt.Circle( (0,0), 0.3, color='white')
plt.pie(per_class, labels=['N', 'L', 'R', 'A', 'V'], colors=['tab:purple','tab:olive','tab:grey','tab:pink','tab:green'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
#plt.title("Sampling in train data")
plt.show()
```

### Step 19: #LSTM Model
#LSTM Model

```python
train_df.shape #sampled training data shape
```

### Step 20: #LSTM model plot
#LSTM model plot

```python
per_class = train_df[train_df.shape[1]-1].value_counts()
per_class #checking the sampled train data class distribution

```

### Step 21: #KFold Cross Validation
#KFold Cross Validation

```python
train_x, test_x, train_y, test_y = train_test_split(rr_intervals, qrs_durations, test_size=0.2, random_state=42)
```

### Step 22: ## LSTM Train Evaluation
## LSTM Train Evaluation

```python
test_x= test.iloc[:,:test.shape[1]-1].values
test_x=test_x.reshape(len(test_x), test_x.shape[1],1) 
test_y= test[test.shape[1]-1] 
test_y=to_categorical(test_y)
```

### Step 23: ## LSTM Test Evaluation
## LSTM Test Evaluation

```python
print("test_x : ", np.shape(test_x)) #independent variables of test data
print("test_y  : ", np.shape(test_y)) #dependent variables of test data
```

### Step 24: #GRU Model
#GRU Model

```python
per_class = test[test.shape[1]-1].value_counts()
print(per_class)
# plt.figure(figsize=(5,5))
# #visualize test data class distribution
# my_circle=plt.Circle( (0,0), 0.3, color='white')
# plt.pie(per_class, labels=['N', 'S', 'V', 'F', 'Q'], colors=['tab:purple','tab:olive','tab:grey','tab:pink','tab:green'],autopct='%1.1f%%')
# p=plt.gcf()
# p.gca().add_artist(my_circle)
# plt.title("No sampling in test data")
# plt.show()
```

### Step 25: ##GRU Model Plot
##GRU Model Plot

```python
target_train=train_df[train_df.shape[1]-1] #label train
#target_val=val[val.shape[1]-1] #labes test
train_y=to_categorical(target_train)
#val_y=to_categorical(target_val)
print(np.shape(train_y))
```

### Step 26: ## KFold cross Validation
## KFold cross Validation

```python
train_x = train_df.iloc[:,:train_df.shape[1]-1].values
#val_x = val.iloc[:,:val.shape[1]-1].values
train_x = train_x.reshape(len(train_x), train_x.shape[1],1)
#val_x = val_x.reshape(len(val_x), val_x.shape[1],1)
print(np.shape(train_x))
```

### Step 27: ### GRU Train Evaluation
### GRU Train Evaluation 

```python
print("train_x : ", np.shape(train_x)) #independent variables in test data
print("train_y  : ", np.shape(train_y)) #dependent variable in test data
#print("val x : ", np.shape(val_x))
#print("val y  : ", np.shape(val_y))
```

### Step 28: ## GRU Test Evaluation
## GRU Test Evaluation

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
import sklearn.metrics as metrics
```

## Conclusion
By following this tutorial, you should have a good understanding of how to process ECG data and apply neural networks for arrhythmia classification. Experiment with the models, adjust their parameters, and try other architectures to further enhance performance.

## Resources
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [Keras Documentation](https://keras.io/)
- [Python Official Documentation](https://docs.python.org/3/)
