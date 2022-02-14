import tensorflow as tf
import keras
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report
import wandb
from wandb.keras import WandbCallback
from data_visualize_util import show_performance, history_plot, confusion_matrix

from model_config import load_model
''''
Description:
'''

def train_model(model_path, hp_path, training_set, val_set, test_set=None, 
                max_epochs=None, classes=None, eval=False, run_wandb=True):

    callbacks_list = [
        # keras.callbacks.ModelCheckpoint(filepath='best_model_'+'-{val_accuracy:.2f}.h5',
        #                                  monitor='accuracy',mode='max', save_best_only=True), 
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',restore_best_weights=True,
                                      verbose=1, patience=5, min_delta=0.001)]
  
    # Load model and hp:
    model = load_model(path=model_path)
    with open(hp_path, 'rb') as f:
        hp = pickle.load(f)
    print('hp:', hp.values)
    # check if use wandb:
    if run_wandb:
        run = wandb.init(project="Semi-Har",
                        group='full_train_stats',
                        config=hp.values)
        callbacks_list.append(WandbCallback())
        
    history = model.fit(training_set[0], training_set[1], 
                        batch_size=hp.values['batch_size'],
                        epochs=max_epochs, # TODO: change for final test
                        callbacks=callbacks_list, 
                        validation_data=val_set, 
                        verbose=1)
    history_plot(history)

    if eval==True and test_set!=None:
        print('Evaluating dataset:')
        y_pred_test = model.predict(test_set[0])
        best_class_pred_test = np.argmax(y_pred_test, axis=1)
        best_class_test = np.argmax(test_set[1], axis=1)
        
        print('Classification report for test data')
        print(classification_report(best_class_test, best_class_pred_test))
        confusion_matrix(best_class_test, best_class_pred_test, classes, normalize=True)
        # Evaluation score: categorical cross-entropy and accuracy
        score = model.evaluate(test_set[0], test_set[1])
        for metric, s in zip(model.metrics_names,score):
            print(f'{metric}: {s*100:.2f}%')

    return model
    
 
            