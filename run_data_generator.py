import os
import argparse
import datetime
import json
import pandas as pd
from unicodedata import name
import requests
import zipfile
from tqdm import tqdm
from sys import get_asyncgen_hooks
from venv import create
from webbrowser import get
from processing_raw_data import *
__author__ = "Marcus Notø"

'''

INFO ABOUT FILE data_generator
For raw processing see file processing_raw_data    
'''
ORIGINAL_DATASET_PATH = 'original_data'
PROCESSED_DATASET_PATH = 'processed_data'

def get_parser():
    parser = argparse.ArgumentParser(
        description='Downloading HAR datasets from public databases and processing datasets')
    parser.add_argument('--working_directory', default='test_runs',
                        help='Directory for where the download and processed datasets are stored')
    parser.add_argument('--mode', default='process', 
                        choices=['download', 'process', 'download_and_process'],
                        help='Which mode to run for the script.\ndownload: download the dataset(s).\nprocess: process the donwloaded dataset(s)')
    parser.add_argument('--dataset', default='MHEALTH', 
                        choices=['PAMAP2', 'MHEALTH', 'all'], 
                        help='name of the dataset to be downloaded/processed')
    parser.add_argument('--data_directory', default='', 
                        help='the path to the downloaded and processed dataset. Default download path is used when None.')
    return parser

def create_paths(args):
    working_directory = args.working_directory

    # Create paths
    if not os.path.exists(working_directory):
        os.mkdir(working_directory)

    dataset_directory =  os.path.join(working_directory, args.data_directory)
    if not os.path.exists(dataset_directory):
        os.mkdir(dataset_directory)
    # Orginal_PATH
    dataset_directory_orginal = os.path.join(dataset_directory, ORIGINAL_DATASET_PATH)
    if not os.path.exists(dataset_directory_orginal):
        os.mkdir(dataset_directory_orginal)
    # Processed PATH
    dataset_directory_processed = os.path.join(dataset_directory, PROCESSED_DATASET_PATH)
    if not os.path.exists(dataset_directory_processed):
        os.mkdir(dataset_directory_processed)
    return working_directory, dataset_directory, dataset_directory_orginal, dataset_directory_processed


def download_data(dataset_directory, dataset_meta):
    data_name = dataset_meta['name']
    data_page = dataset_meta['dataset_home_page']
    data_url = dataset_meta['source_url']
    file_name = dataset_meta['file_name']
    message = f""" '{data_name}' dataset is now being downloaded.
    Please verify that you have visited the homepage of the dataset
    (link: {data_page}) and are familiar with the spesifications.
    Please enter 'y' if agree to be responsible for the download and use of the dataset. 
    """
    answer = input(message)
    if answer == 'y':
        print(os.path.join(dataset_directory, data_name))
        if not os.path.exists(os.path.join(dataset_directory, data_name)):
            os.mkdir(os.path.join(dataset_directory, data_name))
        print("Donwloading ...")

        response = requests.get(data_url, allow_redirects=True, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(os.path.join(dataset_directory, data_name, file_name), 'wb') as f: # where to save the file
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

            progress_bar.close()
            # f.write(response.content) # can use only this if dont want 
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
            raise ValueError('Total size in bytes is not equal to zero when downloading.')
            
        print(f"Finshed donwloading to ({os.path.join(dataset_directory, data_name, file_name)})")


    else:
        print('Canceling downloading since user did not agree to terms.')

def unzip_dataset(dataset_directory, dataset_meta):
    data_name = dataset_meta['name']
    zip_folder = dataset_meta['default_folder_path']
    file_name = dataset_meta['file_name']
    print(f'Unzipping Dataset {data_name}...')
    zip_folder = os.path.join(dataset_directory, data_name, zip_folder)
    if os.path.isdir(zip_folder):
        print('Dataset already unzipped. Exiting..')
        return

    file_path_zip = os.path.join(dataset_directory, data_name, file_name)
    with zipfile.ZipFile(file_path_zip, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(dataset_directory, data_name))
    print('Done unzipping dataset')

def process_dataset(data_path_orginal, data_path_processed,  dataset_meta):
    data_name = dataset_meta['name']
    file_name = dataset_meta['file_name']
    unziped_folder = dataset_meta['default_folder_path']
    print(f'Processing Dataset {data_name}...')
    org_data_path = os.path.join(data_path_orginal, data_name, unziped_folder)
    path_pickle = os.path.join(data_path_processed, data_name+'_processed.pkl')
    if data_name == 'PAMAP2':
        df = PAMAP2_process_files(org_data_path, dataset_meta)
    elif data_name == 'MHEALTH':
        df = MHEALTH_process_files(org_data_path, dataset_meta)
    else:
        print(f'Dataset {data_name} has no raw processing function.')
        print(f'Exiting program.')
        return 
    # save file
    df.to_pickle(path_pickle)
    print(f'Dataset {data_name} is processed an saved to {path_pickle}')



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    working_directory, dataset_directory, dataset_directory_orginal, dataset_directory_processed = create_paths(args)
    # get mode
    mode = args.mode
    # Dataset info:
    with open('DATASET_META.json') as json_file:
        DATASET_METADATA = json.load(json_file)
    # Download section
    if args.mode == 'download' or args.mode == 'download_and_process':
        # check which dataset:
        if args.dataset == 'all':
            datasets = list(DATASET_METADATA.keys())
        else:
            datasets = [args.dataset]
        
        for dataset in datasets:
            try:
                print("Initiating downloading of chosen dataset\n")
                print(f"-------- Downloading {dataset} --------")
                download_data(dataset_directory_orginal, DATASET_METADATA[dataset])
            except Exception as e:
                print('Failed to download dataset: ', e)
        print('Done Downloading')

    if args.mode == 'process' or args.mode == 'download_and_process':
        # TODO: Do some data processing depending on the dataset
        if args.dataset == 'all':
            datasets = list(DATASET_METADATA.keys())
        else:
            datasets = [args.dataset]
        
        for dataset in datasets:
            unzip_dataset(dataset_directory_orginal, DATASET_METADATA[dataset])
            process_dataset(dataset_directory_orginal, dataset_directory_processed, DATASET_METADATA[dataset])
        
        print('Finished generating datasets.')
