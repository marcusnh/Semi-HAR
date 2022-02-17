from cProfile import label
from cgi import test
import os
import glob
import argparse
from click import Path
from matplotlib.font_manager import json_dump
import numpy as np
import matplotlib.pyplot as plt
import modin.pandas as pd
from modin.config import Engine
import datetime
import pickle
import json
import time
from simplejson import load
import tensorflow as tf
import ray
from model_config import create_DNN_model, load_model
from train_model import train_model

from pre_processing_util import * #extract_windows, map_activity_intensity, extract_user_values
from tune_model import run_tuner
__author__ = "Marcus Notø"

'''

INFO ABOUT run_SSL.py

This file is used to create the Semi-Supervised learning(SSL) model for the 
hypersension case. It will also contain the possibilitie to train a supervised
model which will be used as a base case to validate the performance of the 
SSL model.
'''
WINDOW_SIZE = 50 # 50*1/20 =2,5 sec
OVERLAP = 0.5
MODELS_DIR = 'models'
# LABEL_DICT = {
#     'low_int': ['lying', 'sitting','standing', 'watching TV', 'computer work',
#                 'car driving', 'ironing', 'folding laundry', 'frontal elevation of arms',
#                  'waist bends forward'],
#     'medium_int': ['walking', 'nordic walking',"ascending stairs","descending stairs",
#      'vacuum cleaning','house cleaning', 'crouching'],
#     'high_int': ['running', 'cycling', 'playing soccer', 'rope jumping', 'jumping',
#     ]
# }
LABEL_DICT = {
    'low_int': [['lying', 'sitting','standing', 'watching TV', 'computer work',
                'car driving', 'ironing', 'folding laundry', 'frontal elevation of arms',
                 'waist bends forward'],0],
    'medium_int': [['walking', 'nordic walking',"ascending stairs","descending stairs",
     'vacuum cleaning','house cleaning', 'crouching'],1],
    'high_int': [['running', 'jogging', 'cycling', 'playing soccer', 'rope jumping', 'jumping',
    ],2]
}

def get_parser():
    parser = argparse.ArgumentParser(
        description=' Creating DL model with supervised or semi-supervised models')
    parser.add_argument('--working_directory', default='test_runs',
                        help='Directory containing datasets, trained models and training logs')
    parser.add_argument('--mode', default='SL_model', 
                        choices=['SL_model', 'SSL_model'],
                        help='Which mode to run for the script.\SL_model: Create a supervise model for labelled dataset \
                            .\SSL_model: Semi-supervised learning system for the unlabelled dataset.')
    parser.add_argument('--config', default='experiments/test_model.json',
                        help='Congif files for training of differnt DL models.')
    parser.add_argument('--labelled_dataset', default='PAMAP2', type=str,
                        choices=['PAMAP2', 'MHEALTH', None], 
                        help='Name of the labelled dataset for training and fine-tuning')
    parser.add_argument('--unlabelled_dataset', default='MHEALTH', type=str, 
                        choices=['PAMAP2', 'MHEALTH', None], 
                        help='Name of the unlabelled dataset to self-training and self-supervised training, ignored if only supervised training is performed.')
    # TODO: add an argument for WandB if needed
    parser.add_argument('--dataset', default="PAMAP2", 
                        choices=['PAMAP2', 'MHEALTH'], 
                        help='name of the dataset to be downloaded/processed')
    parser.add_argument('-v','--verbose', default=1, type=int,
                    help='verbosity level')

    return parser

def pre_processing(data_path, dataset_meta,  train_size=0.8, test_size=0.5):
    # Get data:
    print(f'Loading data from {data_path}')
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    # Create new activity labels - Map old labels to intensity labels
    ## TODO: add verbose or something so this function becomes optional
    df, label_map = map_activity_intensity(df, dataset_meta['label_list'], LABEL_DICT)
    # Normalise data between 0-1:
    df = normalize_data(df, dataset_meta['sensor_labels'])
    # Create user_list format {user_id: {sensor_vales, activity label}} 
    # # This function is maybe not needed since we are using pandas. 
    user_dict = extract_user_values(df, dataset_meta["sensor_labels"])
    # Extract window: MaybeDO: transfer to dict first. Will make it 10x faster.
    segments_array, labels = extract_windows(df, window_size=WINDOW_SIZE, overlap=OVERLAP)
    # Partition data -  Split data into 3 datasets and perform one-hot encoding:
    np_train, np_val, np_test = split_data(segments_array, labels, 
                                 train_size, test_size) 
    
    input_shape = np_train[0].shape[1:]
    output_shape = np_train[1].shape[1:]
    return {'train': np_train,
        'val': np_val,
        'test': np_test,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'label_map': label_map}

def get_dataset_path(dataset_name, dataset_dir):
    path = glob.glob(dataset_dir +"/"+dataset_name+"*", recursive=True)
    print(path)
    if not path:
        print(f'There exsits no dataset in path:\n {path}')
        print(f'Need to get data from file run_data_generator.py')
        return None
    else:
        return path[0]

def get_datasets(args, DATASET_METADATA, working_directory):
    '''
    This function is used to extract the labelled and unlabelled datasets.
    The preprocessing is happening in function pre_processing.

    INPUT: 
    OUTPUT:
    '''
    datasets = {}
    dataset_dir = os.path.join(working_directory,'processed_data')

    for i, dataset in zip(['labelled', 'unlabelled']
                ,[args.labelled_dataset, args.unlabelled_dataset]):
        print(i, dataset)

        dataset_path = get_dataset_path(dataset,dataset_dir)
        print(f'Pre-processing {i} dataset {dataset}')
        datasets[i] = pre_processing(dataset_path, DATASET_METADATA[dataset],
                                             train_size=0.8, test_size=0.5)
        # TODO: remove down to print:
        pre_processed_path = os.path.join(working_directory, 'pre_processed_data')
        print("pre_processed_path: ", pre_processed_path)
        if not os.path.exists(pre_processed_path):
            os.mkdir(pre_processed_path)
        file = pre_processed_path+'/'+DATASET_METADATA[dataset]['default_folder_path']+'.pkl'
        with open(file, 'wb') as f:
            pickle.dump(datasets[i],f)
        #
        print(f'Done pre-processing dataset {dataset}')

    print('Done pre-processing all datasets.')




def check_if_model_exsists(run_tag, run_type, models_dir):
    paths =[]
    if run_type == 'SL_model':
        model_path = glob.glob(models_dir +"/*"+run_tag+"*.hdf5", recursive=True)
        hp_path = glob.glob(models_dir +"/*"+run_tag+"*.pkl", recursive=True)
        paths.append(model_path[0])
        paths.append(hp_path[0])

    elif run_type == 'Full_SSL_model':
        model_path = glob.glob(models_dir +"/"+run_tag+".hdf5", recursive=True)
        hp_path = None
        paths.append(model_path[0])

    if not (model_path or hp_path):
        print(f'No model for this experiment.')
        print(f'Need to create model before it can be evaluated.')
        return None  
    if len(model_path)>1:
        print('Several similar models. Choosing newest model.')
   
   
    return paths

def mix_datasets(datasets):


    return 

###################################################################################
# Basic DL Steps:
# 1) Analyse and pre-processing data
#   - Choose mode
# 2) Seperate training and test data set 
# 3) Create model / estimator
# 4) Train model and evaluate
# 5) Evaluate model against test data
###################################################################################

if __name__ == '__main__':

    print('Starting Experiment')
    Engine.put("ray")
    ray.init()
    start_time = time.time()
    current_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = get_parser()
    args = parser.parse_args()
    # extact input args:
    working_directory = args.working_directory
    verbose = args.verbose
    dataset = args.dataset
    mode = args.mode
    print('mode: ', mode)
    
    # Dataset info:
    with open('DATASET_META.json') as json_file:
        DATASET_METADATA = json.load(json_file)
    
    models_dir = os.path.join(working_directory, MODELS_DIR)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # Load datasets and do necessary preprocessing:
    datasets = get_datasets(args, DATASET_METADATA, working_directory)

    
                    
    # print(datasets['labelled'])
    # #TODO: remove 
    # with open('Data/PAMAP2_pre_processed.pkl', 'wb') as f:
    #    pickle.dump(datasets['labelled'],f)

    with open('Data/PAMAP2_pre_processed.pkl', 'rb') as f:
        datasets['labelled'] = pickle.load(f)

    #################################################
    print('Loading model configurations and starting experiments.')   
    # TODO: Load configuration
    with open(args.config, 'r') as f:
        config_file = json.load(f)
    exp_tag = config_file['tag']
    experiments = config_file['experiment_configs']

    for run in experiments:
        type = run['type']
        tag = f"{current_time_string}_{type}_{run ['tag']}"
        prev_model = run['tag']
        best_model_path = os.path.join(models_dir,f'{tag}_best.hdf5')
        run['model_path'] = best_model_path
        run['tag'] = tag
        start_m = f'Running experiment {type} with {prev_model}'
        print('\n'+'#'*len(start_m))
        print(start_m)
        print('#'*len(start_m))
        
        # Load model
        if type =='SL_model':
            # STEP 1
            print(f'Creating a SL_model from labelled dataset {dataset}.')
            tune = run['hp_tune']
            # TODO: choose if used only labelled data or unlabelled data for tuning
            # If one eants to tune the model with hyperparameter tuning algorithm
            if tune:
                print(f'Creating new model and starts hyperparamter tuning')
                hp = run['hyperparameters']

                model, best_hp = run_tuner(create_DNN_model, 
                                       datasets['labelled']['train'],
                                       datasets['labelled']['val'],
                                       hp=hp,
                                       run_wandb=True)

                
                model.save(best_model_path)
                print(best_hp)
                path = os.path.join(models_dir,f'{tag}_best.pkl')
                with open(path, 'wb') as f:
                        pickle.dump(best_hp, f)

            # Train the model with optimised hyperparameters and longer epochs
            if run['full_train']:
                # evaluate model
                # Check if model exists
                paths = check_if_model_exsists(prev_model, type, models_dir)
                if paths == None:
                    print(f'Exiting given experiment {type}')
                    continue
                    # TODO: run a create model function?
                else:
                    model_path = paths[0]
                    hp_path = paths[1]
                print(f'Training old SL_model on Dataset {dataset}')
                model = train_model(model_path, hp_path,
                            datasets['labelled']['train'],
                            datasets['labelled']['val'],
                            datasets['labelled']['test'],
                            datasets['labelled']['label_map'].keys(),
                            max_epochs=10,
                            eval=True,
                            run_wandb=True
                            )
                model.save(os.path.join(models_dir,'full_train_'+dataset+'.hdf5'))


        elif type == 'Full_SSL_model':
            # TODO: Check if model exists and if not create model. 
            print('Checking if a teacher model exists')
            # Check if model exists
            print(run['teacher_name'])
            paths = check_if_model_exsists(run['teacher_name'], type, models_dir)
            if paths == None:
                continue
            else:
                model_path = paths[0]
                print(f'Loading model: {model_path}')
                teacher_model = load_model(path=model_path)
            teacher_model.summary()


            # STEP 2: create a mix dataset from labelled dataset D and unlabelled
            # dataset U after removing labels from dataset D. This crates dataset B
            # TODO: create function mix_dataset
            dataset_u = mix_datasets(datasets)
            # STEP 3: Pesudo-labelling of dataset B.
            # Two possibilities:
                # 1) Simple self-labeling: run teacher model.predict on dataset B 
                # and label every window with a score higher than treshold T.
                # Every labelled window goes into a new dataset S.
                # TODO: Create function and give back dataset S and prosentage from
                # dataset D
                
                # 2) Autoencoder self-labeling: Create a autoencoder model from the 
                # unlabelled dataset U. Cluster the entire dataset B and use the 
                # teacher model on each cluster. Every Sample in the cluster with 
                # a precision higher than K(<T) is labelled and inserted into
                #  dataset S.
            # STEP 4: augmentation afrom signal transformaation
                # Adding nosie, scaling by a random scaler, 3D rotation, inverting signal
                # reversing direction of time, random scrambling or stretching.
                # combine samples into a larger dataset D'
            # STEP 5:Train a Student model on Dataset D' and fine tune the model
            # STEP 6: Evaluate on dataset D for confirmation.
            break
        elif type == 'SSL_model':
            print('Run whole process (All steps)')
        else:
            print(f'Experiment {type} is a unknow experiment')
            print(f'Experiment describtion: {tag}')
            print('Continuing to next experiment.')
            continue


    print('Runtime:')
    print("--- %s seconds ---" % (time.time() - start_time))


    
