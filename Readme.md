# Disaster Response Pipeline Project

This project uses data set containing real messages that were sent during disaster events. This project contains a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

### Here are some Screen Shots
![SS](https://github.com/Hnmani/Disaster-response-pipeline/blob/master/data/screencapture-0-0-0-0-3001-go-2020-05-15-21_15_30.png)
Predictor classifying input messages into different categories
![Visual](https://github.com/Hnmani/Disaster-response-pipeline/blob/master/data/Screenshot_20200515_211456.png)
Visualization of the given above.
>>>>>>> d62a29c9bfebbe892ca472b3c360b4a661cd112a

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Components
Project Components
There are three components you'll need to complete for this project.

### 1. ETL Pipeline
In a Python script, `process_data.py`, data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

### 2. ML Pipeline
In a Python script, `train_classifier.py`, ML pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

### 3. Flask Web App
In a Python script, `run.py`, Flask Web App that:
* Visualizes the data
* Displays the output
