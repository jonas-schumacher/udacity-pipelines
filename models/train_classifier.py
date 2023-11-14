import sys

import nltk
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from helper.tokenizer import tokenize

nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

RANDOM_SEED = 42
TABLE_NAME = 'disaster_messages'


def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name=TABLE_NAME, con=engine)
    X = df.iloc[:, 1].values[:100]
    Y = df.iloc[:, 4:].values[:100]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def build_model():
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(tokenizer=tokenize, token_pattern=None)),
        # combines CountVectorizer + TfidfTransformer
        ("classifier", MultiOutputClassifier(RandomForestClassifier(random_state=RANDOM_SEED))),
    ])

    possible_params = pipeline.get_params()

    # TODO: try other params
    parameters = {
        # "vectorizer__ngram_range": ((1, 1), (1, 2)),
        "classifier__estimator__n_estimators": [50, 100, 200],
        # "classifier__estimator__min_samples_split": [2, 3, 4],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    score_dict = {}
    for i, col_name in enumerate(category_names):
        average_scores = classification_report(
            y_true=Y_test[:, i],
            y_pred=Y_pred[:, i],
            output_dict=True,
            zero_division=np.nan,
        )["weighted avg"]
        # print(f"{col_name}: {average_scores}")
        score_dict[col_name] = average_scores
    score_df = pd.DataFrame(score_dict)
    print(score_df.mean(axis=1))
    score_df.to_csv("score_df.csv", sep=";")


def save_model(model, model_filepath):
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

        # Test tokenize function
        # print(tokenize(X[0]))

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    database_filepath = "../data/DisasterResponse.db"
    model_filepath = 'disaster.pkl'

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

    # Test tokenize function
    # print(tokenize(X[0]))

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

    # main()
