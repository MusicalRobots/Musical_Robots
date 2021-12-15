# Background: 

We want users to learn more about the music they like.  Given a music file, we will predict the genre of the song. 
We will also return the most popular songs in that genre and other similar genres the user may be interested in. 

# User Profiles
## User 1:
User has an audio file.  User wants to know the genre of the music file. The user must be able to run the command "streamlit run file" from the terminal. 

The user uploads their music file, and uses the Musical Robot UI to predict the genre, learn the most popular song in the genre, and learn what the other most similar genres are.  The user can choose whether to listen to their uploaded file, the most popular song, or a random song from a similar genre. 

## User 2:
User wants to predict the genre for a batch of audio files using the pre-trained Support Vector Machine and return an accuracy report.  The user wants to be able to query the datasets.  User knows how to operate a Jupyter Notebook.

The user can follow the tutorial in GenrePredictionTutorial.  The user will create dataframes and the training, testing, and validation datasets using the functions in spectrogram_dataset.  The user will run the pre-trained SVM using the functions in svm_prediction and return an accuracy report if they are itnerested.  The user can use the functions in dataset_queries to query the created dataframes.

## User 3:
The user has a fundamental understanding of machine learning and wants to train their own Support Vector Machine for music genre prediction.  The user knows how to operate a Jupyter Notebook

The user can follow the tutorial in TrainSVMTutorial.  They will be able to use the functions in spectrogram_dataset to curate their own datasets for training, testing, and validation.  They will be able to run their SVM and return an accuracy report using the functions in svm_prediction.

# Data Sources:

FMA: A Dataset For Music Analysis
  - tracks.csv per track metadata such as ID, title, artist, genres, tags, and play counts 
  - genres.csv all genres present in data
  - fma_small 8000 tracks of 30s, 8 balanced genres 

