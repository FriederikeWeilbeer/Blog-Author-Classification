import hashlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
import time
import re
import os
import multiprocessing as mp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import random

import joblib

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix

import seaborn as sns

from nltk.probability import FreqDist

STOP_WORDS = set(stopwords.words('english'))
START_TIME = time.time()
LAST_STAMP = time.time()


def main():
    """
    Main function
    """

    log("Start")
    training_pipeline("gender",
                      COMPLETE_DATA_LENGTH,
                      test_split_percentage=0.3,
                      shuffle=True,
                      shuffle_state=RANDOM_STATE,
                      evaluate=True,
                      evaluate_dist=True,
                      overwrite=False)

    log("Finished program")


def training_pipeline(column,
                      max_rows,
                      test_split_percentage=0.3,
                      shuffle=False,
                      shuffle_state=random.randint(10000, 20000),
                      evaluate=False,
                      evaluate_dist=False,
                      overwrite=True
                      ):
    """
    Function for the training pipeline

    :param column: column of dataframe which should be trained
    :param max_rows: max rows for import
    :param test_split_percentage: percentage of test data
    :param shuffle: flag for shuffling data
    :param shuffle_state: state to reproducing the shuffle process
    :param evaluate: flag if the model should be evaluated (default: False)
    :param evaluate_dist: flag if a distribution should be generated (default: False)
    :param overwrite: flag for overwriting existing data (default: True)
    """

    # get the state of the method parameters, used for saving the trained model later
    state = str(locals())

    # Step 1: Import data
    data = import_data(FILE_PATH, max_rows)

    # Step 2: Preprocess data
    data = preprocess_data_multiprocessing(data)

    # Step Optional: Analyse Distribution
    if evaluate_dist:
        analyse_distribution(data, ['age', 'gender', 'sign'])

    # Step 3: Prepare data with label
    data = prepare_data_with_label(data, column)

    # Step 4: Split data
    xtrain, xtest, ytrain, ytest = split_training_data(data, test_split_percentage, shuffle_state, shuffle)

    # Step 5: Train Model
    model = train_model(xtrain, ytrain, state, overwrite)

    # Step 6: Evaluate Model
    if evaluate:
        evaluate_model(model, xtest, ytest)


def log(message):
    """
    Function for logging a message with timestamp

    :param message: message for log entry
    """
    global LAST_STAMP

    if LOGGING:
        print(f'[{time.strftime("%H:%M:%S", time.localtime())}]: {message}  -  Execution took {time.time() - LAST_STAMP:.2f}s. Total: {time.time()-START_TIME:.2f}s', )
    LAST_STAMP = time.time()


def import_data(file_path, rows):
    """
    Function for importing data from csv

    :param file_path: path of the csv file
    :param rows: amount of rows to import
    :return: pandas dataframe
    """
    data_frame = pd.read_csv(file_path, delimiter=',', nrows=rows, encoding='utf-8', on_bad_lines='skip')

    log("Finished importing data")

    return data_frame


def analyse_distribution(data_frame, columns):
    """
    Function analysing distribution of values in a column of a dataframe

    :param data_frame: dataframe to analyse
    :param columns: an array of columns to analyse
    """
    for column in columns:
        if column in data_frame.columns:
            build_distribution_graph(data_frame, column)
        else:
            raise ValueError("Column '{}' is not in the dataframe".format(column))


def build_distribution_graph(data_frame, column):
    """
    Function for building a distribution graph

    :param data_frame: dataframe to analyse
    :param column: a column to analyse
    """

    # Get the frequency distribution of every value
    freq = FreqDist(np.array([x for x in data_frame[column]]).ravel())

    # set the figure size
    plt.figure(figsize=(13, 7))

    # Plot bar chart
    bars = plt.bar(freq.keys(), freq.values(), edgecolor='black')  # Adjust bins as needed

    # Add absolute values above the bars
    for bar in bars:
        yval = bar.get_height()

        # Values placing the text
        rotation = 'horizontal'
        text = None

        # Calculate the percentage of one bar
        percentage = (yval / sum(freq.values()) * 100)

        # if the chart contains less than 15 x-values, the text is displayed horizontal
        if len(freq.keys()) < 15:
            rotation = 'horizontal'

            # put text in two lines
            text = f"{round(yval, 2)}\n({round(percentage, 2)}%)"

            # add a margin for text above the bars by extending the y-axis by 1/7
            plt.ylim(top=max(freq.values()) + max(freq.values()) / 7)

        # else it is displayed vertical
        else:
            rotation = 'vertical'

            # put text in one line
            text = f"{round(yval, 2)} ({round(percentage, 2)}%)"

            # add a margin for text above the bars by extending the y-axis by 1/3
            plt.ylim(top=max(freq.values()) + max(freq.values()) / 3)

        # display text
        plt.text(bar.get_x() + bar.get_width() / 2, yval + max(freq.values()) / 50, text, ha='center', va='bottom',
                 rotation=rotation)

    # # Add labels and title to diagram
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.title(column + ' distribution')

    # save the plot to a file
    plt.savefig("graphs/" + column + "_distribution.png", bbox_inches='tight')
    plt.close()


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

    # Replace all special characters with a whitespace to ensure that words are still split
    # f.e. data point #681283 "Hey everybody...and Susan"
    text = re.sub(r"[^\w\s]", " ", text)

    # Replace all multiple whitespaces with a single whitespace
    text = re.sub(r"^\s+", " ", text.strip()).strip()

    # Remove stopwords
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(tokens)


def preprocess_data_multiprocessing(data_frame):
    """
    Function for preprocessing data on multiple cores
    using amount of available cores - 2

    :param data_frame: a pandas dataframe
    :return: preprocessed dataframe
    """

    # get the amount of available cpus
    cpu_count = mp.cpu_count()-2

    # split the dataframe into length/cpu_count dataframes
    df_split = split_dataframe_into_chunks(data_frame, cpu_count)

    # df_split = np.array_split(data_frame, cpu_count) - deprecated

    # create pool with cpu_count amount of threads
    with mp.Pool(cpu_count) as p:

        # concat the split dataframes to a result dataframe and apply preprocess_data function on every split dataframe
        df = pd.concat(p.map(preprocess_data, df_split))

    log("Finished preprocessing data")

    return df


def split_dataframe_into_chunks(data_frame, chunk_size):
    """
    Function for splitting a pandas array into equal chunks
    Used because numpy.array_split() uses deprecated function

    :param data_frame: a pandas dataframe
    :param chunk_size: number of chunks to split the dataframe into
    :return: split dataframe
    """

    # Calculate the number of rows in each part and the remainder
    rows_per_part, remainder = divmod(len(data_frame), chunk_size)

    # Split the DataFrame into equal parts
    df_split = [data_frame.iloc[i * rows_per_part:(i + 1) * rows_per_part] for i in range(chunk_size)]

    # Distribute remaining rows among the parts
    for i in range(remainder):
        df_split[i] = pd.concat([df_split[i], data_frame.iloc[chunk_size * rows_per_part + i:i + 1 + chunk_size * rows_per_part]])

    return df_split


def prepare_data_with_label(data_frame, column):
    """
    Function for preparing data
    by creating a dataframe with text and labels as columns

    :param data_frame: a pandas dataframe
    :param column: the column name with the values
    :return: a dataframe with text and labels
    """
    result_data_frame = data_frame[["text", column]].copy()
    result_data_frame.columns = ['text', 'label']

    # todo: comment
    result_data_frame["label"] = result_data_frame["label"].astype(str)

    log("Finished preparing data with label")

    return result_data_frame


def split_training_data(data_frame, test_split_percentage, shuffle_state, shuffle):
    """
    Function for splitting data into training and testing sets

    :param data_frame: a pandas dataframe
    :param test_split_percentage: percentage of the test data
    :param shuffle_state: state to reproducing the shuffle process
    :param shuffle: flag for shuffling
    :return: xtrain, xtest, ytrain, ytest
    """
    x = data_frame["text"]
    y = data_frame["label"]

    log("Finished splitting data")

    return train_test_split(x.values, y.values, test_size=test_split_percentage,
                            random_state=shuffle_state, shuffle=shuffle)


def train_model(x_train, y_train, state_string, overwrite):
    """
    Function for training the model with LogisticRegression

    :param x_train: a pandas dataframe with all training data
    :param y_train: a pandas dataframe with all labels
    :param state_string: a string containing the state of the model
    :param overwrite: a flag to overwrite existing models
    :return: xtrain, xtest, ytrain, ytest
    """
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

    # Generate String representing the configuration of the model
    pipeline_str = repr(model) + state_string

    # Calculate the hash value out of model state for the model name
    # used to load and save models with the same state
    hash_value = hashlib.sha256(pipeline_str.encode()).hexdigest()

    # file path of the saved model
    file_path = "serialized/" + hash_value + ".joblib"

    # if the model exists and overwrite is disabled
    if os.path.exists(file_path) and not overwrite:
        model = joblib.load(file_path)

    # if the model does not exist, or overwrite is enabled
    else:
        model.fit(x_train, y_train)
        joblib.dump(model, file_path)

    log("Finished training the model")
    return model


def evaluate_model(model, x_test, y_test):
    """
    Function for evaluating the model with test data

    :param model: a trained model
    :param x_test: the test values which should be predicted
    :param y_test: the correct labels of the test datat
    """

    y_pred = model.predict(x_test)

    # generate a dataframe containing predictions and correct values
    df = pd.DataFrame()
    df["text"], df["correct"], df["prediction"] = x_test, y_test, y_pred
    df["right_prediction"] = np.where(df["correct"] == df["prediction"], 1, 0)

    # print scores of the model
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred, average="macro"))
    print("Precision: ", precision_score(y_test, y_pred, average="macro"))
    print("F1-Score: ", f1_score(y_test, y_pred, average="macro"))

    # get labels of the data for plot
    unique_labels = sorted(set(y_test))

    # set plot size
    plt.figure(figsize=(16, 9))

    # get confusion matrix of the model
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # generate a heatmap out of the confusion matrix
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # save the heatmap to a file
    heatmap.get_figure().savefig('confusion_matrix_heatmap.png', bbox_inches='tight')
    plt.close()

    log("Finished evaluating the model")


FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284
RANDOM_STATE = 41236451
LOGGING = True

# driver
if __name__ == "__main__":
    main()

