
import pandas as pd
import numpy as np

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