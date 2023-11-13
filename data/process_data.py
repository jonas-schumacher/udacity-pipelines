import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    # ### 1. load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # ### 2. Merge datasets.
    # - Merge the messages and categories datasets using the common id
    # - Assign this combined dataset to `df`, which will be cleaned in the following steps

    df = pd.merge(messages, categories, on="id", how="inner")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # ### 3. Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names.

    categories = df["categories"].str.split(pat=";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # ### 4. Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # ### 5. Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)

    # ### 6. Remove duplicates.
    # - Check how many duplicates are in this dataset.
    # - Drop the duplicates.
    # - Confirm duplicates were removed.
    print(f"Duplicates before dropping: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Duplicates after dropping: {df.duplicated().sum()}")

    return df


def save_data(df: pd.DataFrame, database_filename: str):
    # ### 7. Save the clean dataset into an sqlite database.

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(name='disaster_messages', con=engine, if_exists="replace", index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    messages_filepath = "disaster_messages.csv"
    categories_filepath = "disaster_categories.csv"
    database_filepath = "DisasterResponse.db"

    print('Loading data...')
    df = load_data(messages_filepath=messages_filepath, categories_filepath=categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print("Saving data...\n    DATABASE: {}".format(database_filepath))
    save_data(df, database_filepath)

    # main()
