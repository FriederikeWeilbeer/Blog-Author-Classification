import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import time
import re
import multiprocessing as mp

STOP_WORDS = set(stopwords.words('english'))


def main():
    """
    Main function
    """
    now = time.time()
    data = import_data(FILE_PATH, 5000)

    print(data['text'].apply(len).mean())
    data = preprocess_data_multiprocessing(data)
    print(data['text'].apply(len).mean())

    data = prepare_data_with_label(data, "age")
    print(data)

    print(f'Execution took {time.time() - now:.2f} seconds', )


def import_data(file_path, rows):
    """
    Function for importing data from csv

    :param file_path: path of the csv file
    :param rows: amount of rows to import
    :return: pandas dataframe
    """
    data_frame = pd.read_csv(file_path, delimiter=',', nrows=rows, encoding='utf-8', on_bad_lines='skip')
    return data_frame


def preprocess_data(data_frame):
    """
    Function for preprocessing text column of a dataframe

    :param data_frame: a pandas dataframe
    :return: preprocessed dataframe
    """
    data_frame['text'] = data_frame['text'].apply(clean_text)
    return data_frame


def clean_text(text):
    """
    Function for cleaning text by replacing special characters and stopwords

    :param text: text to be cleaned
    :return: cleaned text
    """

    # Remove special chars
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"^\s+", " ", text.strip())

    # Remove stopwords
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(tokens)


def preprocess_data_multiprocessing(data_frame):
    """
    function for preprocessing data on multiple cores
    using amount of available cores - 2

    :param data_frame: a pandas dataframe
    :return: preprocessed dataframe
    """

    # get the amount of available cpus
    cpu_count = mp.cpu_count()-2

    # split the dataframe into length/cpu_count dataframes
    df_split = np.array_split(data_frame, cpu_count)

    # create pool with cpu_count amount of threads
    with mp.Pool(cpu_count) as p:

        # concat the split dataframes to a result dataframe and apply preprocess_data function on every split dataframe
        df = pd.concat(p.map(preprocess_data, df_split))
    return df


def prepare_data_with_label(data_frame, column):
    """
    function for preparing data
    by creating a dataframe with text and labels as columns

    :param data_frame: a pandas dataframe
    :param column: the column name with the values
    :return: a dataframe with text and labels
    """
    result_data_frame = data_frame[["text", column]].copy()
    result_data_frame.columns = ['text', 'label']
    return result_data_frame


FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284

# driver
if __name__ == "__main__":
    main()
