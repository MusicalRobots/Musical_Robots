import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn as skl
import sklearn.utils
import IPython.display as ipd
from sklearn.model_selection import train_test_split
import ast
import os
from pathlib import Path


def load_test():
    #check that the data can load, (getting an error in Windows)
    try:
        #read input file
        file_paths = pd.read_csv('data/all_data_paths.txt', header = None, names = ['file_path'])

        #find and load some file
        index = 7999
        filename = 'data/fma_small/' + file_paths['file_path'][index]
        y, sr = librosa.load(filename, sr=None, mono=True)
    except:
        #fails is any error pops up
        print("load_test failed.")
        return
    
    #passed when successfully ran all code
    print("load_test_passed.")
    return 


#run all the tests when file is ran
load_test()
