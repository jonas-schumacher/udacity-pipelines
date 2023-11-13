import re
import sys

import nltk
import numpy as np
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

RANDOM_SEED = 42

# ### 1. Load data from database.
# - Define feature and target variables X and Y

engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table(table_name="disaster_messages", con=engine)
X = df.iloc[:, 1].values[:100]
y = df.iloc[:, 4:].values[:100]
col_names_output = list(df.columns[4:])


# ### 2. Write a tokenization function to process your text data

def tokenize(text):
    # Normalize text
    normalized_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(normalized_text)

    # Remove stop words
    words_cleaned = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    words_lemmatized = [lemmatizer.lemmatize(w) for w in words_cleaned]
    words_lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in words_lemmatized]

    return words_lemmatized


def tokenize_from_flask(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Test tokenize function
# print(tokenize(X[0]))

# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=tokenize)),  # combines CountVectorizer + TfidfTransformer
    ("classify", MultiOutputClassifier(RandomForestClassifier(random_state=RANDOM_SEED))),
])

# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED)
pipeline.fit(X_train, y_train)

# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.
y_pred = pipeline.predict(X_test)
score_dict = {}
for i, col_name in enumerate(col_names_output):
    average_scores = classification_report(
        y_true=y_test[:, i],
        y_pred=y_pred[:, i],
        output_dict=True,
        zero_division=np.nan,
    )["weighted avg"]
    # print(f"{col_name}: {average_scores}")
    score_dict[col_name] = average_scores
score_df = pd.DataFrame(score_dict)
print(score_df.mean(axis=1))
score_df.to_csv("score_df.csv", sep=";")

# ### 6. Improve your model
# Use grid search to find better parameters.

parameters = None

cv = None

# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.
#
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# ### 9. Export your model as a pickle file
dump(pipeline, 'disaster.pkl')


def load_data(database_filepath):
    pass


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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

# if __name__ == '__main__':
#     main()
