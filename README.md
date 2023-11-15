# Disaster Response Pipeline Project

This project classifies text messages using NLP preprocessing and Scikit-Learn ML Pipelines

## Code Overview:

The code is separated into 3 parts:

1. ETL Pipeline
    - (E) Load Data from CSV files, (T) Clean Data and (L) Save to Database
    - data/process_data.py
2. ML Pipeline
    - Preprocess input text using NLP techniques + Train MultiOutputClassifier
    - model/train_classifier.py
3. Web Application
    - Serve trained model in a Flask web application
    - app/run.py

### Instructions:

1. Run ETL pipeline that cleans data and stores in database:
   `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. Run ML pipeline that trains classifier and saves
   `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run Flask Web App, go to `app` directory and run `python run.py`