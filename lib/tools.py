import os
import json
import csv
import pandas as pd


def make_dir(dir):
    '''
    from string or list of strings
    create directory
    '''
    if isinstance(dir, list):
        for d in dir:
            os.makedirs(d, exist_ok=True)
    else:
        os.makedirs(dir, exist_ok=True)



def LoadFile_list(path):
    '''
    '''
    with open(path) as file:
        data = file.read().splitlines()    
    return(data)



def save_json(path, data):
    '''
    '''
    with open(path, 'w') as outfile:
        json.dump(data, outfile)



def load_json(file_path):
    '''
    '''
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data




def load_csv(path):
    '''
    '''
    data = []
    with open(path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(dict(row))
    return data


################################################################################
# DataFrames
def save_pickle(path, pd):
    '''
    '''
    pd.to_pickle(path)



def load_pickle(path):
    '''
    '''
    data = pd.read_pickle(path)
    return(data)



def df_to_csv(path, df):
    '''
    '''
    df.to_csv(path, index=False)
    


def df_to_json(path, df):
    '''
    '''
    df.to_json(path, orient='records')
    


def df_load_csv(path):
    '''
    '''
    df = pd.read_csv(path)
    return(df)