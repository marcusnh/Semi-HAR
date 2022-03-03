from cProfile import label
from cgi import test
import os
import sys
import shutil
import gc
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
from pandas import array
from simplejson import load
import tensorflow as tf
import ray
from SSL_tune_model import run_tuner
from sklearn.utils import compute_class_weight, shuffle
from sklearn.model_selection import train_test_split

from SSL_model_config import create_DNN_model, load_model
from SSL_train_model import train_model, evaluate_model
from SSL_utilities import *
from pre_processing_util import * #extract_windows, map_activity_intensity, extract_user_values

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
# LABEL_INTENSITY_DICT = {
#     'low_int': ['lying', 'sitting','standing', 'watching TV', 'computer work',
#                 'car driving', 'ironing', 'folding laundry', 'frontal elevation of arms',
#                  'waist bends forward'],
#     'medium_int': ['walking', 'nordic walking',"ascending stairs","descending stairs",
#      'vacuum cleaning','house cleaning', 'crouching'],
#     'high_int': ['running', 'cycling', 'playing soccer', 'rope jumping', 'jumping',
#     ]
# }
LABEL_INTENSITY_DICT = {
    'low_int': [['lying', 'sitting','standing', 'watching TV', 'computer work',
                'car driving', 'ironing', 'folding laundry', 'frontal elevation of arms',
                 'waist bends forward'],0],
    'medium_int': [['walking', 'nordic walking',"ascending stairs","descending stairs",
     'vacuum cleaning','house cleaning', 'crouching'],1],
    'high_int': [['running', 'jogging', 'cycling', 'playing soccer', 'rope jumping', 'jumping',
    ],2]
}

NEW_LABEL_DICT = {'standing': [['Standing'],0],
                'lying': [['lying'],1],
                'sitting': [['sitting','watching TV', 'car driving', 'computer work'],2],
                'walking': [['walking','nordic walking'],3],
                'walking up/down stairs': [['descending stairs','ascending stairs'],4],
                'running': [['running','jogging'],5],
                'jumping': [['jumping','rope jumping'],6],
}

def get_parser():
    parser = argparse.ArgumentParser(
        description=' Creating DL model with supervised or semi-supervised models')
    parser.add_argument('--working_directory', default='test_runs',
                        help='Directory containing datasets, trained models and training logs')
    
    parser.add_argument('--config', default='experiments/test_model.json',
                        help='Congif files for training of differnt DL models.')
    parser.add_argument('--labelled_dataset', default='MHEALTH', type=str,
                        choices=['PAMAP2', 'MHEALTH', None], 
                        help='Name of the labelled dataset for training and fine-tuning')
    parser.add_argument('--unlabelled_dataset', default='PAMAP2', type=str, 
                        choices=['PAMAP2', 'MHEALTH', None], 
                        help='Name of the unlabelled dataset to self-training and self-supervised training, ignored if only supervised training is performed.')
    # TODO: add an argument for WandB if needed
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

    df, label_map = map_activity_intensity(df, dataset_meta['label_list'], 
                                            LABEL_INTENSITY_DICT)
  
    ## TODO: balance data:
    # activity_occurrence = count_activities(df)

    # Normalise data between 0-1:
    df = normalize_data(df, dataset_meta['sensor_labels'])

    # Create user_list format {user_id: {sensor_vales, activity label}} 
    # # This function is maybe not needed since we are using pandas. 
    # user_dict = extract_user_values(df, dataset_meta["sensor_labels"])
    # Extract window: MaybeDO: transfer to dict first. Will make it 10x faster.
    segments_array, labels = extract_windows(df, window_size=WINDOW_SIZE, overlap=OVERLAP)
    
    # Partition data -  Split data into 3 datasets and perform one-hot encoding:
    np_train, np_val, np_test = split_data(segments_array, labels, 
                                 train_size, test_size) 
   
    activity_count = dict(zip(*np.unique(labels, return_counts=True)))

    input_shape = np_train[0].shape[1:]
    output_shape = np_train[1].shape[1:]
   
    return {'train': np_train,
        'val': np_val,
        'test': np_test,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'label_map': label_map,
        'activity_count': activity_count}

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
        pre_processed_path = os.path.join(working_directory, 'final_data')
        print("pre_processed_path: ", pre_processed_path)

        if not os.path.exists(pre_processed_path):
            os.mkdir(pre_processed_path)

        file = pre_processed_path+'/'+DATASET_METADATA[dataset]['default_folder_path']+'.pkl'
        with open(file, 'wb') as f:
            pickle.dump(datasets[i],f)
        #
        print(f'Done pre-processing dataset {dataset}')

    print('Done pre-processing all datasets.')
    return datasets

def load_data():
    '''
    Helper function. 
    TODO: Should not be in final code. maybe remove
    '''
    datasets = {}
    with open('test_runs/processed_data/MHEALTHDATASET.pkl', 'rb') as f:
        datasets['labelled'] = pickle.load(f)
    with open('test_runs/processed_data/PAMAP2_Dataset.pkl', 'rb') as f:
        datasets['unlabelled'] = pickle.load(f)
    return datasets


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

def generate_mixed_datasets(datasets, dataset_name = 'mix_unlabelled'):

    if (datasets['labelled']['input_shape'] != datasets['unlabelled']['input_shape'] or
        datasets['labelled']['output_shape'] != datasets['unlabelled']['output_shape'] or
        datasets['labelled']['label_map'] != datasets['unlabelled']['label_map']):
        print('The datasets are not compatible. The program will exit the experiment.')
        exit(0)
    datasets[dataset_name] = {}
    # create new mixed_dataset
    # TODO: maybe remove labelled data
    for set in ['train', 'val', 'test']:
        x_l = datasets['labelled'][set][0]
        x_u = datasets['unlabelled'][set][0]
        x_c = np.concatenate((x_l, x_u))
        shuffle = np.random.permutation(len(x_c))
        # Extract the labels just for testing
        y_l = datasets['labelled'][set][1]
        y_u = datasets['unlabelled'][set][1]
        y_c = np.concatenate((y_l, y_u))
        np_value = (x_c[shuffle], y_c[shuffle])
        datasets[dataset_name][set] = np_value
     # Add additional dataset info   
    datasets[dataset_name]['input_shape'] = np_value[0].shape[1:]
    datasets[dataset_name]['output_shape'] = np_value[1].shape[1:]
    datasets[dataset_name]['label_map'] = datasets['labelled']['label_map']

    return datasets
def get_predict_data(dataset):
    # Leave test set out to be able to check performance
    x_data = (dataset['train'][0],dataset['val'][0])
    x_data = np.concatenate(x_data)
    return x_data

def extract_best_class(x_data, y_pred, threshold=0.70, samples_per_class=5000):
    sample_list = np.full(len(x_data), False, dtype=bool)
    nr_classes = y_pred.shape[1]
    print(nr_classes)
    print('Loop through each class and extracting values above threshold:')
    for c in range(nr_classes):
        print(f"---Processing class {c}---")
        samples_in_class = np.full(len(x_data), True, dtype=bool)
        # Plurality test:
        print('Plurality test')
        print((np.argmax(y_pred, axis=1) == c) & samples_in_class)
        samples_in_class = (np.argmax(y_pred, axis=1) == c) & samples_in_class
        print(f"Passes plurality test: {np.sum(samples_in_class)}")
        # Threshold test:
        print('Threshold test')
        samples_in_class = (y_pred[:, c] >= threshold) & samples_in_class
        print(f"Passes minimum threshold: {np.sum(samples_in_class)}")

        if np.sum(samples_in_class) > samples_per_class:
            print('Class has more samples than the max number of allowed')
            # Only set the number of sample per class equal to True, rest False
            ## All values that are not true is sat to 0 and extract index
            y_pred_masked =  np.where(samples_in_class,y_pred[:,c], 0 )
            samples_indicies = np.argpartition(-y_pred_masked, samples_per_class)
            # Only pick the nr of sample we want:
            samples_in_class[samples_indicies[:samples_per_class]] = True
            samples_in_class[samples_indicies[samples_per_class:]] = False
            print(f'Final number of samples for this class:', np.sum(samples_in_class))
            print(f'With minimum confidence : {y_pred[samples_indicies[samples_per_class-1],c]}')

        # Add selected samples to the overall sample list:
        sample_list = samples_in_class | sample_list
        print('Total number of samples:', np.sum(sample_list))

    
    return x_data[sample_list], y_pred[sample_list]

def self_labeling(datasets, teacher_model):
    # TODO: Predict label
    print('Running Selflabeling of the mixed dataset')
    print(teacher_model.summary())
    unlabelled_mix = get_predict_data(datasets['mix_unlabelled'])
    print(unlabelled_mix.shape)
    unlabelled_pred_prob = teacher_model.predict(unlabelled_mix, batch_size=352)

    print(unlabelled_pred_prob)
    print(unlabelled_pred_prob.shape)
    self_labelled_data = extract_best_class(unlabelled_mix, unlabelled_pred_prob, threshold=0.50,)
    print(self_labelled_data[1])
    return self_labelled_data
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
    dataset_l = args.labelled_dataset
    dataset_u = args.unlabelled_dataset
  
    # Dataset info:
    with open('DATASET_META.json') as json_file:
        DATASET_METADATA = json.load(json_file)
    
    models_dir = os.path.join(working_directory, MODELS_DIR)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # Load datasets and do necessary preprocessing:
    datasets = get_datasets(args, DATASET_METADATA, working_directory)
    # datasets = load_data()



    #################################################
    print('Loading experiement configurations.')   
    # Load configuration
    with open(args.config, 'r') as f:
        config_file = json.load(f)
    exp_tag = config_file['tag']
    experiments = config_file['experiment_configs']

    # Run experiments:
    for run in experiments:
        # Clean directory
        gc.collect()
        tf.keras.backend.clear_session()
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
        previous_model = False
        
        # Load model
        if type =='SL_model':
            # STEP 1
            print(f'Creating a SL_model from labelled dataset {dataset_l}.')
            
            # TODO: choose if used only labelled data or unlabelled data for tuning
            # If one eants to tune the model with hyperparameter tuning algorithm
            hp = run['hyperparameters']
            if run['hp_tune']:
                print(f'Creating new model and starts hyperparamter tuning')
                

                model, best_hp = run_tuner(create_DNN_model, 
                                       datasets['labelled']['train'],
                                       datasets['labelled']['val'],
                                       hp=hp,
                                       run_wandb=True,
                                       )

                model.save(best_model_path)
                print(best_hp)
                path = os.path.join(models_dir,f'{tag}_best.pkl')
                with open(path, 'wb') as f:
                        pickle.dump(best_hp, f)
                # Remove wandb folder:
                remove_folder(path='wandb')
                previous_model= True

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
                print()
                print(f'Training old SL_model on Dataset {dataset_l}')
                model_name = run['teacher_name']
                model = train_model(
                            datasets['labelled']['train'],
                            datasets['labelled']['val'],
                            datasets['labelled']['test'],
                            model_path =model_path, 
                            hp_path=hp_path,
                            nr_epochs=10,
                            classes=datasets['labelled']['label_map'].keys(),
                            eval=True,
                            run_wandb=True,
                            hp = hp,
                            name = model_name,
                            existing_model=previous_model)
                            
                model.save(os.path.join(models_dir, model_name+'.hdf5'))
                # Remove wandb folder:
                remove_folder(path='wandb')


        elif type == 'Full_SSL_model':
            # Remove wandb folder:
            remove_folder(path='wandb')
            
            # Check if model exists and if not create model. 
            print('Checking if a teacher model exists')
            print(run['teacher_name'])
            paths = check_if_model_exsists(run['teacher_name'], type, models_dir)
            if paths == None:
                # TODO: Create model or exit experiment
                print(f'There exist no teacher model for labelled dataset {dataset_l}')
                print('Need to run a SL_model experiment with full training.')
                continue
            else:
                model_path = paths[0]
                print(f'Loading model: {model_path}')
                teacher_model = load_model(path=model_path)
            
            if run['test_performance']:

                print(f'Test performance for model on unlabelled data')
                print('This function is only used for testing with labelled data')
                evaluate_model(teacher_model, datasets['unlabelled']['test'])
                count_nr_classes(datasets['unlabelled']['test'][1])
                # scores = teacher_model.evaluate(datasets['unlabelled']['test'][0], 
                #                                 datasets['unlabelled']['test'][1])
                # for metric, s in zip(teacher_model.metrics_names,scores):
                #     print(f'{metric}: {s*100:.2f}%')

            # STEP 2: create a mix dataset from labelled dataset D and unlabelled
            # dataset U after removing labels from dataset D. This crates dataset B
            # create function mix_dataset
            print(f'Mixing labelled dataset {dataset_l} and unlabelled dataset{dataset_u}')
            datasets = generate_mixed_datasets(datasets)
            # STEP 3: Pesudo-labelling of dataset B.
            # Two possibilities:
            # 1) Simple self-labeling: run teacher model.predict on dataset B 
            # and label every window with a score higher than treshold T.
            # Every labelled window goes into a new dataset S.
            # Create function and give back dataset S and prosentage from
            # dataset D
            print(f'Running simple Self-Labeling')
            self_labelled_data = self_labeling(datasets, teacher_model)
            if run['transform_data']:
                print('Create transform function to agumentate data')
            else:
                print('Use predicted labels as training input.')
                print('Split data into training and validation set.')
                X_train, X_val, y_train, y_val = train_test_split(self_labelled_data[0],
                                                self_labelled_data[1],
                                                test_size=0.1, random_state=42)
                self_training_set = (X_train, y_train)
                self_val_set = (X_val, y_val)
            print('Done')
            # TODO: Balance the data:
            # TOOD: function to show the current balance
            # count_nr_classes(self_labelled_data[1])
            # y_data = convert_one_hot_encoding(self_labelled_data[1])
            # print(self_labelled_data[0])
            # print(self_labelled_data[0].shape)
            # print(y_data.shape)
            # x_data = self_labelled_data[0].reshape(-1,4)
            # self_labelled_data = under_sample(x_data, y_data)
            # unique, counter = np.unique(y_data, return_counts=True)
            # print("The class distribution of the data are:")
            # for i, nr in zip(unique, counter):
            #     print(f'Class {i+1}: {nr}')
                  
            

            # TODO
            # 2) Autoencoder self-labeling: Create a autoencoder model from the 
            # unlabelled dataset U. Cluster the entire dataset B and use the 
            # teacher model on each cluster. Every Sample in the cluster with 
            # a precision higher than K(<T) is labelled and inserted into
            #  dataset S.
            # TODO
            # STEP 4: augmentation a from signal transformaation
                # Adding nosie, scaling by a random scaler, 3D rotation, inverting signal
                # reversing direction of time, random scrambling or stretching.
                # combine samples into a larger dataset D'
            # TODO: Class imbalance: balanse the class
            # STEP 5:Train a Student model on Dataset D' and fine tune the model
            #  Testing: Train student model on dataset
            student_name = 'student_self_train_'+dataset_l
            student_model = train_model(self_training_set, self_val_set, 
                                existing_model=False,
                                hp=run["hyperparameters"], 
                                nr_epochs=10,
                                classes=datasets['labelled']['label_map'].keys(),
                                # STEP 6: Evaluate on dataset D for confirmation.  
                                eval=True, 
                                test_set=datasets['unlabelled']['test'],
                                run_wandb=True,
                                name = student_name)
            ## save model:
            student_model.save(os.path.join(models_dir, student_name+'.hdf5'))


            # STEP 6: Evaluate on dataset D for confirmation.                            
            # if run['test_performance']:
            #     print(f'Test performance for model on unlabelled data')
            #     print('This function is only used for testing with labelled data')
            #     evaluate_model(student_model, datasets['unlabelled']['test'])
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


    
