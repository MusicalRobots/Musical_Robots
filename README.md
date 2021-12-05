# Musical_Robots
A project that helps to identify the genre of an mp3 music file and discover other music of the same genre. 


## Repository Structure
- /docs/: Component specification and functional specification.
- /musical_robots/data/: Raw data that was used for training the ML model.
- /musical_robots/Tests/: Unit tests.


## Use
1) Download full repository. 
2) To use the service to identify an mp3 music file's genre please open the following file: 'musical_robots/musical_robots_start.ipynb' and follow the instructions. 
3) To replicate the ML model:  
			- Download the training dataset 'fma_small.zip' at the https://github.com/mdeff/fma.  
			- Open the 'musical_robots/SVMPrediction.py'
4) To run the unit tests, use `python -m unittest -c tests`  at the /musical_robots/ directory.
5) To check out a Neural Networks approach for training the model, please open 'musical_robots/TrainANetwork.ipynb'.



## Contributors
Olga Dorabiala, Katie Johnston, Raphael Liu, Amre Abken.

## Acknowledgements
Thanks to Prof. David Beck and Anant Mittal at the University of Washington for teaching the course and guiding us during the execution of this project. 
