
from pickle import TRUE
from re import S
from zlib import Z_BLOCK
import modin.pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import heartpy as hp
from scipy.signal import resample
import neurokit2 as nk

def downsampling(df,from_hz, to_hz=20):
    df_ds = pd.DataFrame(columns=df.columns)
    down_sized = from_hz/to_hz
    df_ds = df.groupby(np.arange(len(df))//down_sized).mean()
    # Remove overlapping classes: label that is not a whole number
    df_ds = df_ds[df_ds.activity_id % 1  == 0]

    return df_ds

def upsampling(df):
    '''
    Inter polationg between values withn spacing given by
    the index.
    Assuming that there are values in the column and some
    NaN values. Interpolate and remove edges that is not 
    interpolated.
    '''
    df['HR (bpm)'] = df['HR (bpm)'].interpolate(method='index')  

    
    # df.plot(x='timestamp',y=['temp', 'HR (bpm)'])

    return df

def extract_user_values(df, columns):

    users_dict = {}
    users = df['subject_id'].unique()
    for user in users:
        user_data = df[df['subject_id']==user]
        data_values = user_data[columns].values
        labels =  user_data['activity_id'].values
        users_dict[user]=[(data_values,labels)]
    return users_dict


def extract_windows(data, window_size, overlap=0.5):
    '''
    Extract windows from the dataframe
    '''

    
    if overlap>1 or overlap<=0:
        print("Invalid Input Entered")
        print("Overlap_prosent value must be between [0,100]")
        print("And the total overlap can't be zero, must at least be one")
        print('Overlap is sat to the 1/%')
        overlap =0.01
    overlap = window_size - int(window_size*(overlap))
    # user_dict = extract_user_values(data, sensor_columns)

    values = []
    labels = []
    drop_columns =['subject_id', 'activity_id']
    # N_FEATURES = len(data.drop(columns=drop_columns).columns)
    N_FEATURES = 4

    # TODO: finish this
    for i in range(0, len(data)- window_size, overlap):
        label = stats.mode(data['activity_id'].iloc[i: i + window_size])[0][0]
        labels.append(label)
   
        acc_x = data['acc_x'].values[i: i + window_size]
        acc_y = data['acc_y'].values[i: i + window_size]
        acc_z = data['acc_z'].values[i: i + window_size]
        # gyr_x = data['gyr_x'].values[i: i + window_size]
        # gyr_y = data['gyr_y'].values[i: i + window_size]
        # gyr_z = data['gyr_z'].values[i: i + window_size]
        # mag_x = data['mag_x'].values[i: i + window_size]
        # mag_y = data['mag_y'].values[i: i + window_size]
        # mag_z = data['mag_z'].values[i: i + window_size]
        hr_bpm = data['HR (bpm)'].values[i: i + window_size]
        # temp = data['temp'].values[i: i + window_size]


        values.append([acc_x, acc_y, acc_z, hr_bpm])
        # values.append([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y,
        #                 mag_z, hr_bpm, temp ])
  

    # reshape into an array with x rows and columns equal to window_size, and seperate for each feature 
    segments_reshaped = np.asarray(values, 
                        dtype=np.float32).reshape(-1, window_size, 
                        N_FEATURES)
    labels = np.asarray(labels)

    return segments_reshaped, labels


def map_activity_intensity(df, df_labels, new_label_dict):
    '''
    map_activity_intensity function maps the old activity labels to new intensity
    labels ( low, middel and high intensity). This is used if hypersension only
    wants to know how active each subject is and not the exact activity.     
    '''
    label_map = {x:new_label_dict[x][1] for x in new_label_dict.keys()}

    for activity, value in df_labels.items():
        for new_activity in new_label_dict.values():
            if activity in new_activity[0]:
                df.loc[df.activity_id==value, 'activity_id'] = new_activity[1]
                break
    return df, label_map


def normalize_data(data, sensor_columns):
    '''
    normalize_data between [-1,1] for each sensor column.
    Can also use z-normalisation for this. 
    '''
    for column in data[sensor_columns]:
        numerator = (data[column]-data[column].min())
        denominator = (data[column].max()-data[column].min())
        data[column] = 2*(numerator/denominator)-1
        
    return data

def split_data(segments_array, segments_labels, train_size, test_size):
    '''
    Split_data is a function which splits the data into 3 sets (training, 
    validation and test) given the train_size and test_size. The training 
    size indicates how much data should be used for trainig and test_size
    how much of the remaining data should be used for testing.
    The testing data is left out of the training section later on.
    '''
    random_seed =42
    X_train, X_test, y_train, y_test = train_test_split(segments_array, 
        segments_labels, train_size=0.8, random_state = random_seed)

    X_val, X_test, y_val, y_test = train_test_split(X_test, 
        y_test, test_size=0.5, random_state = random_seed)

    y_train_hot= tf.keras.utils.to_categorical(y_train, dtype='uint8')
    y_val_hot= tf.keras.utils.to_categorical(y_val, dtype='uint8')
    y_test_hot= tf.keras.utils.to_categorical(y_test, dtype='uint8')
    np_train = (X_train, y_train_hot)
    np_val = (X_val, y_val_hot)
    np_test = (X_test, y_test_hot)
  
    return np_train, np_val, np_test


def ECG_to_HR(df, sample_rate=50):
    '''
    Inspiration from https://github.com/neuropsychology/NeuroKit/blob/master/docs/examples/heartbeats.ipynb

    '''
    print('Extracting HR from ECG signals')
    print('This may take some time')

    ecg = df[['ECG_1', 'ECG_2']].mean(axis=1).to_numpy()
    # Differnet cleaning methods;
    # signals = pd.DataFrame({"ECG_Raw" : ecg,
    #                          "ECG_NeuroKit" : nk.ecg_clean(ecg, sampling_rate=50, method="neurokit"),
    #                         #  "ECG_BioSPPy" : nk.ecg_clean(ecg, sampling_rate=100, method="biosppy"),
    #                         #  "ECG_PanTompkins" : nk.ecg_clean(ecg, sampling_rate=50, method="pantompkins1985"),
    #                         #  "ECG_Hamilton" : nk.ecg_clean(ecg, sampling_rate=50, method="hamilton2002"),
    #                         #  "ECG_Elgendi" : nk.ecg_clean(ecg, sampling_rate=50, method="elgendi2010"),
    #                         #  "ECG_EngZeeMod" : nk.ecg_clean(ecg, sampling_rate=500, method="engzeemod2012")
    #                          })
    # signals.plot() #doctest: +ELLIPSIS

    # Automatically process the (raw) ECG signal
    ecg_signals, info = nk.ecg_process(ecg, sampling_rate=sample_rate)
    # plot = nk.ecg_plot(ecg_signals, sampling_rate=sample_rate)
    df.insert(loc=len(df.columns)-1, column='HR (bpm)', value=(ecg_signals['ECG_Rate']))
    # Remove ECG signals
    df = df.drop(columns=['ECG_1', 'ECG_2'])
    # # print(df['HR (bpm)'].value_counts())
    # plt.show()

    return df