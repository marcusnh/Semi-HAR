import  os
import time
import modin.pandas as pd
from modin.config import Engine
import ray
import numpy as np
import matplotlib.pyplot as plt
from pre_processing_util import ECG_to_HR, downsampling,upsampling
import cProfile
import re
import json


__author__ = "Marcus Notø"

'''

INFO ABOUT FILE procesing_raw_data.py
File for processing the raw data input from differnt datasets   
'''
OPT_HZ = 20
def PAMAP2_process_files(data_files_path, dataset_meta):
    Engine.put("ray")
    ray.init()
    # For PAMAP2 there are two folders with subject data.
    columns = ['activity_id', 'HR (bpm)',
                 'temp','acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z',
                  'mag_x','mag_y', 'mag_z']
    columns_needed = dataset_meta['columns_needed']
    sub_folders = ['Protocol','Optional']

    # Parse through both subfolders:
    df_list = []
    for sub_folder in (sub_folders):
        for file in os.scandir(os.path.join(data_files_path,sub_folder)):
            if file.is_file():
                df = pd.read_csv(file.path, header=None, sep=' ', engine='python')
                
                # Add wanted columns and names
                df = df[columns_needed]
                df.columns = columns
                # Add subject_iD: 
                subject_id = int(file.path[-7:-4])
                df.insert(loc=1,column='subject_id', value=([subject_id]*len(df)))
                
                # Remove activity_id = 0
                df = df[df.activity_id !=0]

                # Downsampling of IMU to 20 Hz
                df = downsampling(df, dataset_meta['sampling_rate_IMU'],OPT_HZ)
            
                # Upsampling og HR to 20 hz
                df = upsampling(df)

                
                # Reduce memory size with .to_numeric
                int_cols = ['subject_id','activity_id', 'HR (bpm)']
                cols = df.columns.drop(int_cols)
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', downcast="float")
                df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce', downcast="unsigned")
                
                # Drop NaN values
                df = df.dropna()
                # df.to_pickle('test_runs/processed_data/PAMAP2_processed')
                # df = pd.read_pickle('test_runs/processed_data/PAMAP2_processed')
                # print(df)
                # print('done')
                # Done processing raw dat file, append to list:
                df_list.append(df)

    # Combine dataframes:
    PAMAP_df = pd.concat(df_list)
    # PAMAP_df.to_csv('Data/PAMAP2.csv')

    # print meory usage:
    print(PAMAP_df.info(memory_usage="deep"))
    print(PAMAP_df['subject_id'].value_counts())
    print(PAMAP_df)

    # Add data to dictionary and save to pickle:
    # PAMAP_data = PAMAP_df.to_pickle()


    return PAMAP_df

def MHEALTH_process_files(data_files_path, dataset_meta):
    Engine.put("ray")
    ray.init()
    # costume column labels needed
    columns_labels = ['acc_x','acc_y','acc_z', 'ECG_1', 'ECG_2', 'activity_id']
    columns_needed = dataset_meta['columns_needed']
    df_list = []
    for file in os.scandir(data_files_path):
        if file.is_file() and (not '.txt' in file.path):
            print(f'Extracting file {file.path}')
            df = pd.read_csv(file.path, header=None, sep='\t', engine='python')

            # Add wanted columns and names
            df = df[columns_needed]
            df.columns = columns_labels

            

            # Extract HR (bpm) from ECG:
            df = ECG_to_HR(df)
          
            # Add subject_iD: 
            subject_id = int('10'+file.path[-5:-4])
            df.insert(loc=0,column='subject_id', value=([subject_id]*len(df)))
           
            # Remove activity_id = 0
            df = df[df.activity_id !=0]

            # Downsampling of IMU and ECG to 20 Hz
            df = downsampling(df, dataset_meta['sampling_rate'],OPT_HZ)
            # Drop NaN values
            df = df.dropna()

            # Reduce memory size with .to_numeric
            int_cols = ['subject_id','activity_id']
            cols = df.columns.drop(int_cols)
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', downcast="float")
            df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce', downcast="unsigned")
            
          
            # print('done')
            # Done processing raw dat file, append to list:
            df_list.append(df)

    # Combine dataframes:
    MHEALTH_df = pd.concat(df_list)
    MHEALTH_df.to_csv('Data/MHEALTH.csv')

    # print meory usage:
    print(MHEALTH_df.info(memory_usage="deep"))
    print(MHEALTH_df['subject_id'].value_counts())
    print(MHEALTH_df['activity_id'].value_counts())
    print(MHEALTH_df)
    return MHEALTH_df
            



if __name__ == '__main__':
    # When using modin
    with open('DATASET_META.json') as json_file:
        DATASET_METADATA = json.load(json_file)
    print('Run Test')
    start_time = time.time()
    path = 'test_runs/original_data/MHEALTH/MHEALTHDATASET'
    # process data and save to pickle
    MHEALTH_process_files(path, DATASET_METADATA['MHEALTH'])
    print('Runtime:')
    print("--- %s seconds ---" % (time.time() - start_time))
