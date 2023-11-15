import sys

import pandas as pd
from sqlalchemy import create_engine

TABLE_NAME = "disaster_messages"


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load datasets from CSV files and merge to one DataFrame.

    Parameters
    ----------
    messages_filepath: str
    categories_filepath: str

    Returns
    -------
    pd.DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on="id", how="inner")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by extracting columns that will later be used as labels. Drop duplicates.

    Parameters
    ----------
    df: pd.DataFrame
        Input DF

    Returns
    -------
    pd.DataFrame
        Output DF

    """
    # Split categories into separate category columns:
    categories = df["categories"].str.split(pat=";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    # Replace old column by new ones:
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)

    # Values "0" and "2" from column "related" seem to have same meaning --> replace "2" by "0"
    df.loc[df["related"] == 2, "related"] = 0
    # After doing so, output should be binary:
    all(df.iloc[:, 4:].isin([0, 1]))

    # Drop duplicates:
    print(f"Duplicates before dropping: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Duplicates after dropping: {df.duplicated().sum()}")

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    Save the clean dataset into a sqlite database.

    Parameters
    ----------
    df: pd.DataFrame
        cleaned DataFrame
    database_filename: str
        Path to database

    Returns
    -------
    None

    """

    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(name=TABLE_NAME, con=engine, if_exists="replace", index=False)


def main():
    print(f"Arguments provided: {sys.argv[1:]}")
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
