import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
import time
import re
import multiprocessing as mp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import pickle

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix

import seaborn as sns

from nltk.probability import FreqDist

STOP_WORDS = set(stopwords.words('english'))


def main():
    """
    Main function
    """
    now = time.time()
    data = import_data(FILE_PATH, 2000)

    data = preprocess_data_multiprocessing(data)

    analyse_distribution(data)

    #data = prepare_data_with_label(data, "sign")
    #xtrain, xtest, ytrain, ytest = split_training_data(data)

    #model = train_model(xtrain, ytrain)

   # evaluate_model(model, xtest, ytest)

    # Serialize the model to a file using pickle
    #with open('gender.pkl', 'wb') as file:
        #pickle.dump(model, file)

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


def analyse_distribution(data_frame):
    """
    Function for analysing the distribution of data within a data frame

    :param data_frame: a pandas dataframe
    """
    columns = [col for col in data_frame.columns.tolist() if col not in ['text', 'id', 'date']]
    print(columns)
    for column in columns:
        build_distribution_function(data_frame, column)


def build_distribution_function(data_frame, column):
    """
    Function for plotting a distribution function for columns within a data frame

    :param data_frame: a pandas data frame
    :param column: column to be examined
    """
    freq = FreqDist(np.array([x for x in data_frame[column]]).ravel())

    # Plot histogram
    plt.bar(freq.keys(), freq.values(), edgecolor='black')  # Adjust bins as needed

    # # Add labels and title
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.title(column + ' distribution')

    # # Show the plot
    plt.savefig("graphs/" + column + "_distribution.png", bbox_inches='tight')


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
    df_split = np.array_split(data_frame, cpu_count)

    # create pool with cpu_count amount of threads
    with mp.Pool(cpu_count) as p:

        # concat the split dataframes to a result dataframe and apply preprocess_data function on every split dataframe
        df = pd.concat(p.map(preprocess_data, df_split))
    return df


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
    return result_data_frame


def split_training_data(data_frame):
    """
    Function for splitting data into training and testing sets

    :param data_frame: a pandas dataframe
    :return: xtrain, xtest, ytrain, ytest
    """
    x = data_frame["text"]
    y = data_frame["label"]
    return train_test_split(x.values, y.values, test_size=0.3, shuffle=RANDOM_STATE)


def train_model(x_train, y_train):
    """
    Function for training the model with LogisticRegression

    :param x_train: a pandas dataframe
    :param y_train: a pandas dataframe
    :return: xtrain, xtest, ytrain, ytest
    """
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Function for evaluating the model using multiple score, such as Recall, Precision and F1-Score

    :param model:
    :param x_test:
    :param y_test:
    """
    right = 0
    wrong = 0
    y_pred = model.predict(x_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred, average="macro"))
    print("Precision: ", precision_score(y_test, y_pred, average="macro"))
    print("F1-Score: ", f1_score(y_test, y_pred, average="macro"))

    unique_labels = sorted(set(y_test))

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    heatmap.get_figure().savefig('confusion_matrix_heatmap.png', bbox_inches='tight')


FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284
RANDOM_STATE = 41236451

# driver
if __name__ == "__main__":
    main()
