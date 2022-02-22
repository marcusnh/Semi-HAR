import os
import shutil

def remove_folder(path='wandb'):
    try:
        shutil.rmtree(path)
        print(f'Folder {path} was removed')
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    return 