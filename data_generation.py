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

#load data, transform to spectrograms and save to training data

file_paths = pd.read_csv('data/all_data_paths.txt', header = None, names = ['file_path'])
# train_paths, test_paths = train_test_split(file_paths, test_size=0.33) #might want to randomize instead, right now, picking first n
music_data = pd.read_csv('data/fma_metadata/tracks.csv', skiprows = [0,1,2], 
                         usecols = [0, 6, 8, 11, 26, 39, 41, 44, 47, 52], 
                         names = ['track_id', 'album_id', 'album_listens', 'album_title', 'artist_name',
                                 'track_favorites', 'track_genres', 'track_interest', 'track_listens', 'track_title'])
music_data['track_genres'] = music_data['track_genres'].apply(ast.literal_eval)


#get spectrograms and save training data
items_to_keep = 100000 #length to keep, might want to make this longer, but also takes much longer
training_points = 50 #number training points, will need to make this bigger later
labels = np.zeros(training_points)
train_data = np.zeros((training_points, items_to_keep))
for index in range(training_points): #just looks at top 100 files
    if index > training_points:
        break
    print(index)
    filename = 'data/fma_small/' + file_paths['file_path'][index]

    y, sr = librosa.load(filename, sr=None, mono=True)
    train_data[index] = y[:items_to_keep]
    print('Duration: {:.2f}s, {} samples'.format(y.shape[-1] / sr, y.size))

    song_id = file_paths['file_path'][index].rsplit('/')[1].rsplit('.')[0].lstrip('0')
    associated_genres_numeric = music_data[music_data['track_id'] == int(song_id)]['track_genres'].item()
    
    labels[index] = associated_genres_numeric[0] #right now only keeping first label, will need to make it keep all labels later
    
#save training data and labels
np.save('data/train_data.npy', train_data)
np.save('data/train_labels.npy', labels)