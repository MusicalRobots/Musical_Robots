# Major Software Components
1) Data Manager: Provides functionality to load the data as pandas dataframes, split the data into training, test, and validation sets, create a custom audio feature dataset for ML model training, and query subsets of the data.  

2) ML Model: Runs a pre-trained Support Vector Machine on the test data and provides an accuracy report.

2) User Interface: Implementation of the musical robot, which the user can interact with.  The user can upload an mp3 file and follow the prompts given by the robot to predict the genre of the audio and return information about the most popular song in the genre and songs in similar genres.


## Components of Data Manager
<b> Name: </b> create_dataframes

	What it does: Creates dataframes for music audio analysis.

	Inputs: file_paths_path: (str) Path to 'all_data_paths.txt' storing the data paths for each sound file.
        tracks_csv_path: (str) Path to 'tracks.csv' containing general track data.
        genre_csv_path: (str) Path to 'genre.csv' containing genre information.
	
	Outputs: file_path_df: (pd.DataFrame) Dataframe storing the data paths for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        relevant_genre_df: (pd.DataFrame) Dataframe storing genre information for downloaded audio.
        genre_df: (pd.DataFrame) Dataframe storing genre information for all genres in the track_df.
	
<b> Name: </b> split_data

	What it does: Split data into 3 datasets for training, testing and validation.

	Inputs: file_path_df: (pd.DataFrame) Dataframe storing the data paths for each sound file.
        test_percentage: (float) Percentage of paths to designate as part of the test dataset.
        validation_percentage: (float) Percentage of paths to designate as part of the validation dataset.

	Outputs: train_df: (pd.DataFrame) DataFrame of data for the training set
        test_df: (pd.DataFrame) Dataframe of data for the testing set
        validation_df: (pd.DataFrame) Dataframe of data for the validation set

<b> Name: </b> AudioFeature
	
	What it does: Create custom dataset of audio features and genre labels.
	
	Inputs: path_to_data: (str)
            path_df: (pd.Dataframe) DataFrame containing file paths.
            music_df: (pd.DataFrame) Dataframe containing music information.
            genre_df: (pd.DataFrame) Dataframe containing genre information.
	
	Outputs: List of data tuples.

	
<b> Name: </b> create_audio_feature_dataset
	
	What it does: Create the custom dataset given the locations of the data.
	
	Inputs: file_path_df: (pd.DataFrame) Dataframe storing the data paths for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        genre_df: (pd.Dataframe) Dataframe storing genre information.
        path_to_data: (str) Path to where audio data is stored.
        test_percentage: (float) Percentage of paths to designate as part of the test dataset.
        validation_percentage: (float) Percentage of paths to designate as part of the validation dataset.
	
	Outupts: Three AudioFeature datasets.
	
<b> Name: </b> dataset_queries

	What it does: File containing queries to play a given song from a filename, return the most similar genres, return the most popular song in a genre, and play a random song from a given genre.
	
	Inputs: Most of the queries require some combination of: 
	genre: (str) String specifying genre to compare to.
        genre_df: (pd.DataFrame) Dataframe storing genre information.
        track_df: (pd.DataFrame) Dataframe storing general track data.
	path_df: (pd.DataFrame) Dataframe storing path information.
        path_to_data: (str) Path to where audio data is stored.
	
	Outputs: The outputs of the desired query. 
	
<b> Interactions to Accomplish Use Cases </b>
1) User calls create_dataframes to create the dataframes for music audio analysis. 
2) User calls create_audio_feature_dataset to return three AudioFeature datasets.  The system calls split_data to split the data into training, test, and validation sets.  The system then calls AudioFeature on each subset of the data to return three AudioFeature datasets.
3) The user can call any function in dataset_queries to query the dataframes created in step 1.

## Comonents of ML Model

<b> Name: </b> svm_prediction
	
	What it does: Return SVM prediction of genre from an audio file.
	
	Inputs: filename (str): path to music mp3 file
        genre_df (pd.DataFrame): Dataframe containing genre information.
        model_filename (str): Path to the model file.

	Outputs: genre (str): Predicted genre of the mp3 file.

<b> Name: </b> svm_accuracy_report

	What it does: Return accuracy report of trained SVM.
	
	Inputs: true_labels: (List[int]) True labels for genre.
        pred_labels: (List[int]) Predicted labels for genre.
	
	Outputs: tp_rate_list: (List[float]) List of true positive rate of predictions per genre.
        fp_rate_list: (List[float]) List of false positive rate of predictions per genre.
        accuracy: (float) Overall accuracy of predicted labels against true labels.

<b> Interactions to Accomplish Use Cases </b>
1) User calls create_dataframes from the Data Manager components to create the dataframes for music audio analysis. 
2) User calls svm_prediction on an audio file to predict the genre.
3) If testing an SVM an many audio files, user can call svm_accuracy_report to return an accuracy report for their model.

## Components of User Interface

<b> Name: Interactive </b>

	What it does: Implementation of interactive class to communicate with user.  Walks the user through music upload, genre prediction, returns the most popular song in the genre, and plays songs in similar genres.
	
	Inputs: mp3 audio file
	
	Ouputs: User interaction.
	
<b> Interactions to Accomplish Use Cases </b>:
1) User uploads an mp3 audio file.
2) User follows along with the musical robot to return desired information.  System calls on components from both the Data Manager and the ML Model to reutrn the information.

# Preliminary Plan:
1) Cretate a dataset for music genre prediction.
2) Train an ML model for music genre prediction.
3) Write queries for the most popular song in a genre and the most similar genres.
4) Implement the UI for the Musical Robot.
