from importlib.resources import path
import tensorflow as tf
import keras
import os
import shutil
import pickle
import numpy as np
from sklearn.metrics import classification_report
import wandb
from wandb.keras import WandbCallback
from data_visualize_util import show_performance, history_plot, confusion_matrix

from SSL_model_config import load_model, create_DNN_model, create_base_model
from SSL_tune_model import run_tuner
from SSL_utilities import class_weights, remove_folder, get_weights
''''
Description:
'''

def train_model(training_set, val_set, test_set=None, model_path=None, 
                hp_path = None, existing_model=True, hp = None,
                nr_epochs=None, classes=None, eval=False, run_wandb=True,
                name = 'test_full_model', ):

    callbacks_list = [
        # keras.callbacks.ModelCheckpoint(filepath='best_model_'+'-{val_accuracy:.2f}.h5',
        #                                  monitor='accuracy',mode='max', save_best_only=True), 
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',restore_best_weights=True,
                                      verbose=1, patience=5, min_delta=0.001)]
  
    # Load model and hp:
    if existing_model:
        
        model = load_model(path=model_path)
        with open(hp_path, 'rb') as f:
            hp = pickle.load(f)
        print('hp:', hp.values)
    else:
        print(f'Creating new model and starts hyperparamter tuning')
        model, hp = run_tuner(create_base_model, 
                                training_set,
                                val_set,
                                hp=hp,
                                run_wandb=run_wandb)
        # Remove wandb folder:
        remove_folder(path='wandb')

    # Start training model with full dataset
    # check if use wandb:
    if run_wandb:
        run = wandb.init(project="Semi-Har",
                        group=name,
                        config=hp.values)
        callbacks_list.append(WandbCallback())
        
    history = model.fit(training_set[0], training_set[1], 
                        batch_size=hp.values['batch_size'],
                        epochs=nr_epochs, # TODO: change for final test
                        callbacks=callbacks_list, 
                        validation_data=val_set, 
                        class_weight=get_weights(training_set[1]),
                        verbose=1)
    history_plot(history)

    if eval==True and test_set!=None:
        evaluate_model(model,test_set, classes=classes)
    else:
        print('The model was not evaluated on a Test set')
   
    return model
    
 
def evaluate_model(model, test_set, classes=None ):
    print('Evaluating dataset:')
    y_pred_test = model.predict(test_set[0])
    best_class_pred_test = np.argmax(y_pred_test, axis=1)
    best_class_test = np.argmax(test_set[1], axis=1)
    print('Classification report for test data')
    print(classification_report(best_class_test, best_class_pred_test,target_names=classes))
    if classes is not None:
        print('Confusion matrix:')
        confusion_matrix(best_class_test, best_class_pred_test, classes, normalize=True)
    # Evaluation score: categorical cross-entropy and accuracy
    score = model.evaluate(test_set[0], test_set[1])
    for metric, s in zip(model.metrics_names,score):
        print(f'{metric}: {s*100:.2f}%')
    '''
         # Get metrics
        accuracy = accuracy_score(y_test, model.predict(x_test))
        precision = precision_score(y_test, model.predict(x_test))
        recall = recall_score(y_test, model.predict(x_test))
        f1score = f1_score(y_test, model.predict(x_test))
    '''

    '''
    # ROC plot and precision-recall curve:
    from scikitplot.metrics import plot_roc
    from scikitplot.metrics import plot_precision_recall
    y_score = model.predict_proba(test_set[0])
    # Plot metrics 
    plot_roc(y_test, y_score)
    plt.show()
    plot_precision_recall(y_test, y_score)
    plt.show()  
    '''