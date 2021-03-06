import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the data
    :param messages_filepath: message filepath must be .csv
    :param categories_filepath: categories filepath must be .csv
    :return: merged Dataframe
    """
    message = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(message, categories, on = ['id'])
    return df


def clean_data(df):
    """
    This function cleans the data. It converts all the categories into columns
    drop duplicates.
    :param df: dataframe to be cleaned
    :return: cleaned dataframe
    """
    categories = df['categories'].str.split(';',expand = True)
    categories.columns = list(map(lambda x: x.split('-')[0].strip(), categories.loc[0]))
    for cols in categories.columns:
        categories[cols] = categories[cols].str.split('-').str[1]
    df = pd.concat([df, categories], axis = 1)
    df.drop(columns = ['categories'], inplace = True)

    df.drop_duplicates(subset = ['id'], inplace = True)
    assert (df.shape[0] == len(df['id'].unique()))

    return df


def save_data(df, database_filename):
    """
    saves the dataframe to sqlite database.
    :param df: dataframe to be saved as database
    :param database_filename: filename of the database
    :return: None
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("DisasterResponse", engine, index = False)


def main():
    """
    This is the main function which performs all task by calling the functions also indicates the
    state of execution.
    :return:
    pass
    """
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
