from matplotlib.cbook import print_cycles
import pandas as pd
from modin.config import Engine
import ray
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn import metrics

import json
import time

from pre_processing_util import map_activity_intensity

LABEL_DICT = {
    'low_int': [['lying', 'sitting','standing', 'watching TV', 'computer work',
                'car driving', 'ironing', 'folding laundry', 'frontal elevation of arms',
                 'waist bends forward'],0],
    'medium_int': [['walking', 'nordic walking',"ascending stairs","descending stairs",
     'vacuum cleaning','house cleaning', 'crouching'],1],
    'high_int': [['running', 'jogging', 'cycling', 'playing soccer', 'rope jumping', 'jumping',
    ],2]
}


def total_activities(data):
    sns.set_style('whitegrid')
    sns.color_palette("Paired")
    # data['activity'].value_counts().plot(kind='bar', title='Number of acitivity samples')
    sns.countplot(x='activity_id', data=data, palette='Paired')
    for i, count in enumerate(data['activity_id'].value_counts()):
        print(f'For class {i+1}: {count} ({count/len(data)*100:.1f}%)')
    plt.show()

def sample_extraction(data):
    data = data[0:128]
    fig, axes = plt.subplots(nrows=3, ncols=1,  figsize=(6, 6))
    data = data.drop(columns=['timestamp', 'user_id', 'activity'])
    data['x-axis'].plot(ax = axes[0], c='r')

    axes[0].set_ylabel('Acc x-axis', fontsize=20)
    axes[0].legend(loc='upper right')
    data['y-axis'].plot(ax = axes[1], c='g')

    axes[1].set_ylabel('Acc y-axis', fontsize=20)
    axes[1].legend(loc='upper right')
    data['z-axis'].plot(ax = axes[2], c='b')

    axes[2].legend(loc='upper right')
    axes[2].set_ylabel('Acc z-axis', fontsize=20)

def activity_data_per_user(data):
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Dejavu Sans'

    plt.figure(figsize=(16,8))
    plt.title('Data provided by each user', fontsize=20)
    sns.countplot(x='user_id',hue='activity', data = data)
    plt.show()

def activity_wise_dist(data, column):
    # sns.set_palette("Set1", desat=0.80)
    facetgrid = sns.FacetGrid(data, hue='activity_id', palette='Paired')
    facetgrid.map(sns.kdeplot, column ).add_legend()
    plt.show()
   


def activity_boxplot_dist(data, column):
    #group magnitude data
    sns.set_palette("Set1", desat=0.80)
    # facetgrid = sns.FacetGrid(data, hue='activity')
    # facetgrid.map(sns.kdeplot, column ).add_legend()
    # plt.show()
    #Show boxplot:
    plt.figure(figsize=(7,5))
    sns.boxplot(x='activity', y=column,data=data, showfliers=False, saturation=1)
    plt.ylabel(column +'')
    # plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
    # plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
    plt.xticks(rotation=40)
    plt.show()

def show_activity(data, activity, user_id, start = 0, samples=None):

    # Show single activity for a spesific user
    user_data = data[data['user_id']==user_id]
    user_data = user_data.drop(columns=['timestamp', 'user_id'])
    print('\nNumber of samples per activity:')
    print(user_data['activity'].value_counts())
    user_data = user_data[user_data['activity']==activity]
    if samples == None:
        samples = len(user_data)

    index = range(start, samples)
    new_data = user_data[start:samples]
    new_data.index = index
    print(new_data)
    title = activity+' for user: ' +str(user_id)
    new_data.plot(title=title, )
    plt.legend(loc='upper right')

def plot_activity(data, activity, user_id):

    # Show single activity for a spesific user
    user_data = data[data['subject_id']==user_id]
    user_data = user_data.drop(columns=[ 'subject_id'])
    print('\nNumber of samples per activity:')
    print(user_data['activity_id'].value_counts())
    user_data = user_data[user_data['activity_id']==activity]
    title = activity+' for user: ' +str(user_id)
    user_data.plot(title=title, )
    plt.legend(loc='upper right')


def compare_user_activitys(data, user_id, samples=128, activity=None):
    #Look at all activities  or one activity of user with pre defined number of samples
    user_data = data[data['user_id']==user_id]
    user_data = user_data.drop(columns=['timestamp', 'user_id'])
    activities = []
    if activity == None:
        activities = user_data['activity'].unique()
        print('\nNumber of samples per activity:')
        print(user_data['activity'].value_counts())
    else:
        activities.append(activity)

    fig, axes = plt.subplots(nrows=1, ncols=len(activities), figsize=(10, 5))
    counter = 0
    # user_data[user_data['activity'] == 'Jogging'].to_csv(path_or_buf='testfile.csv')
    for i in activities:
        activity_data = user_data[user_data['activity'] == i]
        activity_data.index = range(0,len(activity_data))
        if activity == None:
            activity_data[0:samples].plot(title=i, ax=axes[counter])
        else:
            activity_data[0:samples].plot(title=i, ax=axes)
        counter +=1
    plt.show()
    

def activity_difference_between_users(data, users, activity, samples=128):
    cnt = 0
    fig, axes = plt.subplots(nrows=1, ncols=len(users), figsize=(20, 10))
    plt.suptitle('Comparing activity:'+activity)
    for i in users:
        user_data =data[(data['user_id'] ==i) & (data['activity'] == activity)]
        user_data =user_data[0:samples]
        user_data.index = range(0,len(user_data))
        # if user_data.empty:
        #     print(user_data)

        user_data = user_data.drop(columns=['timestamp', 'user_id'])
        user_data.plot(title='User id: ' +str(i), ax=axes[cnt])
        cnt += 1

    # plt.show()



def confusion_matrix(vali, predict, LABELS, normalize=False):
    matrix = metrics.confusion_matrix(vali, predict) 
    print('\n ********Confusion Matrix********')
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    fmt = '.2f' if normalize else 'd'
    print(matrix)
    plt.figure( figsize=(6, 4))
    sns.heatmap(matrix, cmap='coolwarm', linecolor='white',linewidths=1, 
                xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=fmt)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def show_performance(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and  Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

def history_plot(hist):
    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax[0].plot(hist.history['loss'], color='b', label="Training loss")
    ax[0].plot(hist.history['val_loss'], color='r', label="Validation loss")
    legend = ax[0].legend(loc='best', shadow=True)
    ax[0].set_xlabel('Training epoch')

    ax[1].plot(hist.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(hist.history['val_accuracy'], color='r',label="Validation accuracy")
    ax[1].set_xlabel('Training epoch')
    ax[1].set_ylim([0.5,1])
    legend = ax[1].legend(loc='best', shadow=True)
    #fig.supxlabel('Training Epoch')
    plt.show()

def execute_TSNE(data, perplexities=[2,5,10,20,50], n_iter=1000, 
                 img_name_prefix='t-sne'):
    y_data = data['activity']
    # print(y_data)
    X_data = data.drop(['user_id', 'activity'],axis=1)
    for i, perplexity in enumerate(perplexities):
        # create new x iwth lower dimensions:
        X_new= TSNE(verbose=2, n_iter=n_iter,
                         perplexity=perplexity).fit_transform(X_data)
        # new dataframe to create plot:
        df = pd.DataFrame({'x':X_new[:,0], 'y':X_new[:,1],
                           'label':y_data})
        sns.lmplot(data=df, x='x', y='y', hue='label',\
                   fit_reg=False, height=8, palette="Set1",)
                #    markers=['^','P','8','s', 'o','*', 'p'])
        plt.title("perplexity : {} and max_iter : {}".format(perplexity, n_iter))
        img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
        print('saving this plot as image in present working directory...')
        plt.savefig(img_name)
        plt.show()
        print('Done with performing tsne with perplexity {} and with {} iterations at max'.format(perplexity, n_iter))
    return 0


def activity_length(df):
    activity = df.iloc[0]['activity']
    length_activities = []
    activities = []
    counter = 0
    for index, row in df.iterrows():
        if row['activity'] == activity:
            counter += 1
        else:
            # print(activity, counter)
            
            activities.append(activity)
            length_activities.append(counter)

            counter = 0
            activity = row['activity']
    print(activities)
    print(length_activities)
    # plt.bar(activities, length_activities)
    sns.set_palette("Set1", desat=0.80)
    df = pd.DataFrame(list(zip(activities, length_activities)), columns=['activity', 'length'])
    print(df)
    print('Average values:')
    print(df.groupby('activity')['length'].mean())
    print('max values:')
    print(df.groupby('activity').max())
    print('Min values:')
    print(df.groupby('activity').min())
    print('Number of segments:')
    print(df.groupby('activity').count())
    #Show boxplot:
    plt.figure(figsize=(7,5))
    sns.boxplot(x=activities, y=length_activities, showfliers=False, saturation=1)
    plt.xticks(rotation=40)
    plt.ylabel('Size of activity sequence')
    plt.title('Activity sequences information')
    plt.show()


def hypersension_process_files(data_files_path, dataset_meta):
    df_list = []
    # TODO: read files in folder
    # with pd.HDFStore(data_files_path) as hdf:
    # # This prints a list of all group names:
    #     print(hdf.keys())
    #     keys = hdf.keys()
    # for i in keys:
    #     print_cycles(i)
    #     if i == '/data':
    #         continue
    #     df = pd.read_hdf(data_files_path, key=i )
    #     print(df)
    store = pd.HDFStore(data_files_path)
    print(store.keys())
    print(store.info())
    # df = pd.DataFrame(np.random.randn(5,6), columns=['A', 'B', 'C',])
    # store.put('/imu', df, )
    print(store.get('/imu'))
    # print(df)
 
if __name__ == "__main__":
    # When using modin
    with open('DATASET_META.json') as json_file:
        DATASET_METADATA = json.load(json_file)
        
    print('Run Test')
    start_time = time.time()
    # path = 'Data/Deltaker_02_01_24h_g2_preprocessed.hdf'
    path = 'test_runs/processed_data/MHEALTH_processed.pkl'
    data = pd.read_pickle(path)
    total_activities(data)
    activity_wise_dist(data,'HR (bpm)')

    activity_wise_dist(data,'acc_y')
    path = 'test_runs/processed_data/PAMAP2_processed.pkl'
    data2 = pd.read_pickle(path)
    total_activities(data2)
    activity_wise_dist(data2,'HR (bpm)')
    activity_wise_dist(data2,'acc_y')



    # process data and save to pickle
    # Engine.put("ray")
    # ray.init()
    # hypersension_process_files(path, DATASET_METADATA['Hypersension'])
    print('Runtime:')
    print("--- %s seconds ---" % (time.time() - start_time))
