
import numpy as np
import os

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History

from keras_tuner import RandomSearch,BayesianOptimization, Hyperband
from sklearn import datasets

from sympy import Q, hyper
import tensorflow as tf
import kerastuner as kt
import wandb
from wandb.keras import WandbCallback
import pickle

from SL_model_config import *

MODEL_PATH = 'wandb'
class Model_Tuner(kt.Tuner):
    '''
    Building new Tuner class for redefining run_trail() method.
    run_trial() method is defined in kerastuner.Tuner (kt.Tuner) class. 
    Therefore we are inheriting it and redefining the run_trial() method as
    we need
    '''

    def run_trial(self, trial, trainX, trainY, testX, testY, hyperparameters, objective):
        objective_name_str = objective
        
        # Add hyperparameters:
        # TODO: use hyperparameter fict to set remaining hp
        if 'batch_size' in hyperparameters.keys() and hyperparameters!=None:
            bs = hyperparameters['batch_size']
            batch_size = trial.hyperparameters.Int('batch_size',min_value = bs['min_value'], 
                                                    max_value = bs['max_value'], 
                                                    step = bs['step'], 
                                                    default=bs['default'])
        
        hp = trial.hyperparameters
        ## create the model with the current trial hyperparameters
        model = self.hypermodel.build(hp)
        print(wandb.util.generate_id())
        ## Initiates new run for each trial on the dashboard of Weights & Biases
        run = wandb.init(project="Semi-Har",
                        group=hyperparameters['name'],
                         config=hp.values)

        ## WandbCallback() logs all the metric data such as
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, verbose=1),
            #    ModelCheckpoint(mdl_path+'best_model.h5')
            ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=1, factor=0.5, min_lr=1e-4), 
            WandbCallback(monitor='val_loss')
        ]
        ## loss, accuracy and etc on dashboard for visualization

        history = model.fit(trainX,
                  trainY,
                  batch_size=batch_size,
                  epochs=hyperparameters['epochs'],
                  validation_data=(testX,testY),
                #   validation_ratio=0.1,
                  callbacks=[WandbCallback()])  

        ## if val_accurcy used, use the val_accuracy of last epoch model which is fully trained
        val_acc = history.history['val_accuracy'][-1]  ## [-1] will give the last value in the list

        ## Send the objective data to the oracle for comparison of hyperparameters
        self.oracle.update_trial(trial.trial_id, {objective_name_str:val_acc})

        ## save the trial model
        # self.save_model(trial.trial_id, model)
    
        ## ends the run on the Weights & Biases dashboard
        run.finish()




def  run_tuner(model, training_data, test_data, hp, run_wandb=True):
    '''
    run_tuner() to test keras hyperparameter tuner for our  fine tuning
    '''

    wandb.login()
    if not os.path.exists('wandb'):
        os.mkdir('wandb')
    # split data
    trainX = training_data[0]
    trainY = training_data[1]
    testX = test_data[0]
    testY = test_data[1]

   

    input_shape = (None, trainX.shape[1],trainX.shape[2])
    print('input_shape:',input_shape)
    objective = 'val_accuracy'
    if run_wandb:
        ## set the objective of tuning algorithm
        objective = 'val_accuracy'
        run_path = os.path.join(MODEL_PATH, hp["name"])
        if not os.path.exists(run_path):
            os.mkdir(run_path)

        ## Instantiate the new Tuner with tuning algorithm and required parameters
        tuner = Model_Tuner(
            oracle=kt.oracles.Hyperband(
                objective=objective,
                max_epochs=5,
                hyperband_iterations=1),
            hypermodel=model,
            directory=run_path)

        tuner.search_space_summary()

        tuner.search(trainX, trainY, testX, testY, 
                    hyperparameters=hp, objective=objective)

    else:
        tuner = Hyperband(create_DNN_model, 
                            objective=objective, 
                            max_epochs=hp['epochs'],
                            hyperband_iterations=1)

        tuner.search(trainX, trainY, epochs=hp['epochs'], 
                        validation_data=(testX,testY))
        print(tuner.results_summary())

    best_hyperparameter = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameter.values)
    
    # evaluate model on given dataset:
    model = tuner.hypermodel.build(best_hyperparameter)
    history = model.fit(trainX,trainY, epochs=5,)
    _, acc = model.evaluate(testX, testY, verbose=0)
    model.summary()
    print('> %.3f'%(acc*100))
    return model, best_hyperparameter




def seed_all(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


if __name__ =='__main__':
    print('start')
    with open('Data/PAMAP2_pre_processed.pkl', 'rb') as f:
        dataset = pickle.load(f)
    hp = {  "name": "SL_model_labelled_tuning",
            'epochs':10, 
            "batch_size":{ "min_value":32,
                            "max_value":512,
                             "step": 32,
                            "default": 62}}
            
    model, best_hp = run_tuner(create_DNN_model, dataset['train'],
                    dataset['val'], hp=hp)
    path = 'test_runs/models/test.h5'
    model.save(path, save_format='h5')
    model_test = keras.models.load_model(path)
    model, best_hp = run_tuner(load_model,  dataset['train'],
                    dataset['val'], hp=hp)