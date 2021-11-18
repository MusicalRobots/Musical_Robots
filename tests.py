import pandas as pd
import librosa
import librosa.display


def load_test():
    # check that the data can load, (getting an error in Windows)
    try:
        # read input file
        file_paths = pd.read_csv('data/all_data_paths.txt', header=None, names=['file_path'])

        # find and load some file
        index = 7999
        filename = 'data/fma_small/' + file_paths['file_path'][index]
        y, sr = librosa.load(filename, sr=None, mono=True)
    except:
        # fails is any error pops up
        print("load_test failed.")
        return
    
    # passed when successfully ran all code
    print("load_test_passed.")
    return

"""
def test_queries():
    try: #both should fail bc k not in (1,11)
        return_similar_genres(genre: 'pop', genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 0)
        
    try: should fail bc k not in (1,11)
        return_similar_genres(genre: 'pop', genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 11)
    
    try: genre should not exist
        return_similar_genres(genre='screeching cats', genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 10)
        return_most_popular_song(genre: 'screeching cats', genre_df, track_df)
        play_random_song_from_genre(genre= 'screeching cats', genre_df, track_df, path_df)
        
    try: should return none bc audio not in dataset
        play_song_from_title(title = '#1', track_df, path_df)
        
        should play 
        play_song_from_title(title= 'This World', track_df, path_df)
        should fail bc title does not exist
        play_song_from_title(title= 'Blah', track_df, path_df)    
    """



#run all the tests when file is ran
load_test()
