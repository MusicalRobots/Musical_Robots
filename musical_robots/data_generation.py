import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import librosa
# import librosa.display
# import sklearn as skl
# import sklearn.utils
# import IPython.display as ipd
# from sklearn.model_selection import train_test_split
# import ast



def generate_data(idx_list=range(100), items_to_keep=100000, data_path='data/all_data_paths.txt', track_file='data/fma_metadata/tracks.csv',  data_file_name='data/train_data.npy', label_file_name='data/train_labels.npy', main_mp3_directory='data/fma_small/'):
    '''
    generates data as spectrograms from mp3 files
    
    inputs:
        idx_list=range(100): 1d np array, array of indeces that will be turned into data
        items_to_keep=100000: number of steps to keep for each file
        data_path='data/all_data_paths.txt': path where all the data paths are stored
        track_file='data/fma_metadata/tracks.csv':  path where the track csv is stored
        data_file_name='data/train_data.npy': where you want to save the data
        label_file_name='data/train_labels.npy': where you want to save the labels
        main_mp3_directory='data/fma_small/': where the directories of mp3 are
    
    outputs:
        Nothing returned
        data saved in 
    '''

    #load data, transform to spectrograms and save to training data

    file_paths = pd.read_csv(data_path, header = None, names = ['file_path'])
    # train_paths, test_paths = train_test_split(file_paths, test_size=0.33) #might want to randomize instead, right now, picking first n
    music_data = pd.read_csv(track_file, skiprows = [0,1,2], 
                             usecols = [0, 6, 8, 11, 26, 39, 41, 44, 47, 52], 
                             names = ['track_id', 'album_id', 'album_listens', 'album_title', 'artist_name',
                                     'track_favorites', 'track_genres', 'track_interest', 'track_listens', 'track_title'])
    music_data['track_genres'] = music_data['track_genres'].apply(ast.literal_eval)


    # #get spectrograms and save training data
    n_points = len(idx_list)
    labels = np.zeros(n_points)
    data = np.zeros((n_points, items_to_keep))
    for i, index_song in enumerate(idx_list): 
        filename = main_mp3_directory + file_paths['file_path'][index_song]

        y, sr = librosa.load(filename, sr=None, mono=True)
        data[i] = y[:items_to_keep]
#         print('Duration: {:.2f}s, {} samples'.format(y.shape[-1] / sr, y.size))

        song_id = file_paths['file_path'][index_song].rsplit('/')[1].rsplit('.')[0].lstrip('0')
        associated_genres_numeric = music_data[music_data['track_id'] == int(song_id)]['track_genres'].item()

        labels[i] = associated_genres_numeric[0] #right now only keeping first label, will need to make it keep all labels later

    #save training data and labels
    np.save(data_file_name, data)
    np.save(label_file_name, labels)
    return

if __name__ == '__main__':
    #generate train data
    generate_data()
    #generate test data
    generate_data(idx_list=range(100, 150), items_to_keep=100000,  data_file_name='data/test_data.npy', label_file_name='data/test_labels.npy')

# #do same thing to make testing data
# items_to_keep = 100000 #length to keep, might want to make this longer, but also takes much longer
# testing_points = 25 #number training points, will need to make this bigger later
# test_labels = np.zeros(testing_points)
# test_data = np.zeros((testing_points, items_to_keep))
# # print('start loop')
# # print(training_points)
# # print(training_points + testing_points)
# for index in range(training_points, training_points + testing_points): #use next 25 files
# #     if index > testing_points:
# #         break
#     print(index)
#     filename = 'data/fma_small/' + file_paths['file_path'][index]

#     y, sr = librosa.load(filename, sr=None, mono=True)
#     test_data[index-training_points] = y[:items_to_keep]
#     print('Duration: {:.2f}s, {} samples'.format(y.shape[-1] / sr, y.size))

#     song_id = file_paths['file_path'][index].rsplit('/')[1].rsplit('.')[0].lstrip('0')
#     associated_genres_numeric = music_data[music_data['track_id'] == int(song_id)]['track_genres'].item()
    
#     test_labels[index-training_points] = associated_genres_numeric[0] #right now only keeping first label, will need to make it keep all labels later
    
# #save training data and labels
# np.save('data/test_data.npy', test_data)
# np.save('data/test_labels.npy', test_labels)