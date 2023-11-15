import json
import os
import sys

import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine

sys.path.append(os.getcwd())
from common.shared import tokenize, TABLE_NAME, PATH_TO_SCORE_TABLE, PATH_TO_DATABASE, PATH_TO_TRAINED_PIPELINE

app = Flask(__name__)

# load data
engine = create_engine(f"sqlite:///{PATH_TO_DATABASE}")
df = pd.read_sql_table(TABLE_NAME, engine)
score_df = pd.read_csv(PATH_TO_SCORE_TABLE, sep=";", index_col=0)

# load model
model = joblib.load(PATH_TO_TRAINED_PIPELINE)


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    related_df = df[df["related"] == 1]
    category_count = df.iloc[:, 5:].sum(axis=0)
    most_common_categories = category_count.sort_values(ascending=False).iloc[:10]
    least_common_categories = category_count.sort_values(ascending=True).iloc[:10]

    mean_scores = score_df.mean(axis=1).iloc[:3]
    f1_sorted = score_df.loc["f1-score", :].sort_values()

    # create visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Pie(labels=["related", "unrelated"], values=[len(related_df), len(df) - len(related_df)])],
            "layout": {
                "title": "Share of related messages",
            },
        },
        {
            "data": [Bar(x=most_common_categories.index, y=most_common_categories.values)],
            "layout": {
                "title": "Most common categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
            },
        },
        {
            "data": [Bar(x=least_common_categories.index, y=least_common_categories.values)],
            "layout": {
                "title": "Least common categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
            },
        },
    ]

    graphs_training = [
        {
            "data": [Bar(x=mean_scores.index, y=mean_scores.values)],
            "layout": {
                "title": "Average metrics across categories",
                "yaxis": {"title": "Value"},
                "xaxis": {"title": "Metric"},
            },
        },
        {
            "data": [Bar(x=f1_sorted.index, y=f1_sorted.values)],
            "layout": {
                "title": "F1 Score for all categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
            },
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    ids_training = ["graph-training-{}".format(i) for i, _ in enumerate(graphs_training)]
    graphJSON_training = json.dumps(graphs_training, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON, ids_training=ids_training,
                           graphJSON_training=graphJSON_training)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3000, debug=True)


if __name__ == "__main__":
    main()
