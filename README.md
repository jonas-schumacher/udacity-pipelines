# Disaster Response Pipeline Project

This project classifies text messages using NLP preprocessing and Scikit-Learn ML Pipelines

## Code Overview:

The code is separated into 3 parts:

1. ETL Pipeline
    - data/process_data.py
2. ML Pipeline
    - model/train_classifier.py
3. Web Application
    - app/run.py

### Instructions:

1. To run ETL pipeline that cleans data and stores in database
   `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run ML pipeline that trains classifier and saves
   `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. To run the Flask Web App, go to `app` directory and run `python run.py`